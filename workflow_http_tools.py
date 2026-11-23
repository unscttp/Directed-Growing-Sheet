"""HTTPS requests to host-approved origins, over DNS-pinned, verified TLS connections.

Credentials are host configuration: ``http_credentials`` maps an approved origin
to headers (for example ``Authorization``) that are added to every request for
that origin. They never appear in tool arguments, the task journal or prompts,
and a model cannot override them.

Responses are made readable for a model:

* the body is decoded with the charset the server declares (``Content-Type``,
  or for HTML a ``<meta charset>``), not always as UTF-8, so pages in GBK,
  Shift_JIS or Latin-1 are not garbled; ``encoding`` names the charset used;
* compressed transfer is not requested (``Accept-Encoding`` is runtime-managed);
  a gzip or deflate body sent anyway is decompressed within the size limit;
* a body larger than ``max_read_bytes`` is cut there and marked ``truncated``
  instead of failing the call;
* ``"format": "text"`` returns an HTML page as its title and visible text,
  without markup, scripts or styles;
* requests carry a ``User-Agent`` and ``Accept`` header unless the caller sets
  them (many sites refuse requests without them), and a response that looks
  like a block or bot challenge gets a ``note`` saying it is not the page.
"""

from __future__ import annotations

import codecs
import http.client
import ipaddress
import re
import socket
import ssl
import zlib
from html.parser import HTMLParser
from urllib.parse import urlencode, urlsplit, urlunsplit

METHODS = {"GET", "HEAD", "POST", "PUT", "PATCH", "DELETE"}
BLOCKED_HEADERS = {
    "host",
    "content-length",
    "transfer-encoding",
    "connection",
    "proxy-authorization",
    "proxy-connection",
    "upgrade",
    "te",
    "accept-encoding",
}
TIMEOUT_SECONDS = 15
MAX_HEADERS = 32
FORMATS = {"raw", "text"}
DEFAULT_HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; directed-growing-sheet/1.0)",
    "Accept": "text/html,application/xhtml+xml,application/json;q=0.9,*/*;q=0.8",
    "Accept-Encoding": "identity",
}
META_CHARSET = re.compile(rb"""<meta[^>]+charset\s*=\s*["']?\s*([A-Za-z0-9_.:-]+)""", re.I)
BLOCK_MARKERS = (
    "captcha",
    "cf-chl",
    "just a moment...",
    "attention required",
    "enable javascript and cookies",
    "access denied",
)
BLOCK_NOTE = (
    "The site appears to block automated requests or to require a browser; this "
    "content is probably not the page itself."
)
HIDDEN_TAGS = {"script", "style", "noscript", "template", "svg", "head"}
BLOCK_TAGS = {
    "p",
    "div",
    "br",
    "li",
    "ul",
    "ol",
    "tr",
    "table",
    "section",
    "article",
    "header",
    "footer",
    "nav",
    "h1",
    "h2",
    "h3",
    "h4",
    "h5",
    "h6",
    "pre",
    "blockquote",
    "dd",
    "dt",
}


class _TextExtractor(HTMLParser):
    """Title and visible text of an HTML page, one block per line."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.title = ""
        self._parts: list[str] = []
        self._hidden = 0
        self._in_title = False

    def handle_starttag(self, tag, attrs):
        if tag == "title":
            self._in_title = True
        elif tag in HIDDEN_TAGS:
            self._hidden += 1
        elif tag in BLOCK_TAGS:
            self._parts.append("\n")

    def handle_startendtag(self, tag, attrs):
        if tag in BLOCK_TAGS:
            self._parts.append("\n")

    def handle_endtag(self, tag):
        if tag == "title":
            self._in_title = False
        elif tag in HIDDEN_TAGS and self._hidden:
            self._hidden -= 1
        elif tag in BLOCK_TAGS:
            self._parts.append("\n")

    def handle_data(self, data):
        if self._in_title:
            self.title += data
        elif not self._hidden:
            self._parts.append(data)

    def text(self) -> str:
        lines = (" ".join(line.split()) for line in "".join(self._parts).splitlines())
        return "\n".join(line for line in lines if line)


def html_to_text(html: str) -> tuple[str, str]:
    """``(title, visible text)`` of an HTML document."""
    parser = _TextExtractor()
    parser.feed(html)
    parser.close()
    return " ".join(parser.title.split()), parser.text()


def _charset(content_type: str, data: bytes) -> str:
    match = re.search(r"charset\s*=\s*[\"']?([A-Za-z0-9_.:-]+)", content_type or "", re.I)
    candidates = [match.group(1)] if match else []
    if data.startswith(codecs.BOM_UTF8):
        candidates.insert(0, "utf-8-sig")
    if "html" in (content_type or "").lower() or not content_type:
        meta = META_CHARSET.search(data[:4096])
        if meta:
            candidates.append(meta.group(1).decode("ascii", "ignore"))
    for name in candidates:
        try:
            return codecs.lookup(name).name
        except LookupError:
            continue
    return "utf-8"


def _decompress(data: bytes, encoding: str, max_bytes: int) -> bytes:
    encoding = (encoding or "identity").strip().lower()
    if encoding in ("", "identity"):
        return data
    if encoding in ("gzip", "x-gzip", "deflate"):
        wbits = 16 + zlib.MAX_WBITS if "gzip" in encoding else zlib.MAX_WBITS
        try:
            decoder = zlib.decompressobj(wbits)
            # Bounded output: a small compressed body cannot expand past the limit.
            return decoder.decompress(data, max_bytes + 1)
        except zlib.error:
            raise ValueError("HTTP response body could not be decompressed.") from None
    raise ValueError(f"Unsupported HTTP content encoding {encoding!r}.")


def origin(url):
    """Return ``(parsed URL, "https://host:port")`` for an acceptable HTTPS URL."""
    if not isinstance(url, str) or any(ord(c) <= 32 or ord(c) >= 127 for c in url):
        raise ValueError("URL must be ASCII with whitespace encoded.")
    parsed = urlsplit(url)
    if (
        parsed.scheme != "https"
        or not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
        or parsed.fragment
        or "\\" in url
    ):
        raise ValueError("Expected HTTPS URL without userinfo or fragment.")
    if not re.fullmatch(r"[A-Za-z0-9.-]+", parsed.hostname):
        raise ValueError("Expected a DNS hostname or IPv4 address.")
    return parsed, f"https://{parsed.hostname.lower()}:{parsed.port or 443}"


def _check_header(name, value) -> None:
    if (
        not isinstance(name, str)
        or not re.fullmatch(r"[A-Za-z0-9-]+", name)
        or name.lower() in BLOCKED_HEADERS
        or not isinstance(value, str)
        or any(ord(c) < 32 or ord(c) > 126 for c in value)
    ):
        raise ValueError(f"Unsafe HTTP header {name!r}.")


class HTTPPolicy:
    def __init__(self, allowed_origins=(), credentials=None) -> None:
        self.origins: set[str] = set()
        for value in allowed_origins:
            parsed, key = origin(value)
            if parsed.path not in ("", "/") or parsed.query:
                raise ValueError("Allowed origins cannot contain a path or query.")
            self.origins.add(key)
        self._credentials: dict[str, dict[str, str]] = {}
        if not isinstance(credentials or {}, dict):
            raise ValueError("http_credentials must map origins to header objects.")
        for value, headers in (credentials or {}).items():
            _, key = origin(value)
            if key not in self.origins:
                raise ValueError(f"Credentials given for an origin that is not allowed: {value}")
            if not isinstance(headers, dict) or not headers:
                raise ValueError("Each credential entry must be a nonempty header object.")
            for name, header_value in headers.items():
                _check_header(name, header_value)
            self._credentials[key] = dict(headers)

    def __repr__(self) -> str:  # never print credential values
        return f"HTTPPolicy(origins={sorted(self.origins)}, credentials=<hidden>)"

    def validate(self, args):
        parsed, key = origin(args["url"])
        if key not in self.origins:
            raise PermissionError("HTTP origin is not permitted by the host.")
        if args["method"] not in METHODS:
            raise ValueError("Unsupported HTTP method.")
        query = args.get("query", {})
        if not isinstance(query, dict) or any(
            type(k) is not str or type(v) not in (str, int, bool) for k, v in query.items()
        ):
            raise ValueError("HTTP query values must be strings, integers or booleans.")
        headers = args.get("headers", {})
        if not isinstance(headers, dict) or len(headers) > MAX_HEADERS:
            raise ValueError("Invalid HTTP headers.")
        host_managed = {name.lower() for name in self._credentials.get(key, {})}
        seen = set()
        for name, value in headers.items():
            _check_header(name, value)
            if name.lower() in seen:
                raise ValueError(f"Duplicate HTTP header {name!r}.")
            if name.lower() in host_managed:
                raise PermissionError(f"HTTP header {name!r} is supplied by the host.")
            seen.add(name.lower())
        if args["method"] in {"GET", "HEAD"} and args.get("body"):
            raise ValueError("GET and HEAD cannot carry a body.")
        if args.get("format", "raw") not in FORMATS:
            raise ValueError("format must be raw or text.")
        return parsed

    def request(self, context, args, max_bytes):
        parsed = self.validate(args)
        _, key = origin(args["url"])
        query = "&".join(
            value for value in (parsed.query, urlencode(args.get("query", {}))) if value
        )
        parsed = parsed._replace(query=query)
        url = urlunsplit(parsed)
        port = parsed.port or 443
        addresses = socket.getaddrinfo(parsed.hostname, port, type=socket.SOCK_STREAM)
        resolved = [ipaddress.ip_address(item[4][0]) for item in addresses]
        if not resolved or any(
            not ip.is_global or ip.is_multicast or ip.is_reserved for ip in resolved
        ):
            raise PermissionError("HTTP target resolved to a non-public address.")
        family, socktype, protocol, _, address = addresses[0]
        context.record_activity("http.connecting", {"url": url, "address": address[0]})
        # Connect to the address that was checked, then verify TLS against the hostname.
        connection = http.client.HTTPConnection(parsed.hostname, port, timeout=TIMEOUT_SECONDS)
        raw = socket.socket(family, socktype, protocol)
        raw.settimeout(TIMEOUT_SECONDS)
        try:
            raw.connect(address)
            connection.sock = ssl.create_default_context().wrap_socket(
                raw, server_hostname=parsed.hostname
            )
            target = parsed.path or "/"
            if parsed.query:
                target += "?" + parsed.query
            headers = {**args.get("headers", {}), **self._credentials.get(key, {})}
            given = {name.lower() for name in headers}
            for name, value in DEFAULT_HEADERS.items():
                if name.lower() not in given:
                    headers[name] = value
            body = args.get("body")
            connection.request(
                args["method"],
                target,
                body=body.encode("utf-8") if body else None,
                headers=headers,
            )
            response = connection.getresponse()
            data = response.read(max_bytes + 1)
            response_headers = response.getheaders()
            header = {name.lower(): value for name, value in response_headers}
            data = _decompress(data, header.get("content-encoding", ""), max_bytes)
            truncated = len(data) > max_bytes
            data = data[:max_bytes]
            content_type = header.get("content-type", "")
            charset = _charset(content_type, data)
            body = data.decode(charset, errors="replace")
            result = {
                "url": url,
                "status": response.status,
                "headers": response_headers,
                "content_type": content_type,
                "encoding": charset,
                "truncated": truncated,
                "body": body,
            }
            if args.get("format") == "text" and (
                "html" in content_type.lower()
                or body.lstrip()[:15].lower().startswith(("<!doctype", "<html"))
            ):
                result["title"], result["body"] = html_to_text(body)
            sample = body[:4000].lower()
            if response.status in {401, 403, 429, 503} or any(m in sample for m in BLOCK_MARKERS):
                result["note"] = BLOCK_NOTE
            context.record_activity(
                "http.response",
                {
                    "url": url,
                    "status": response.status,
                    "bytes": len(data),
                    "encoding": charset,
                    "truncated": truncated,
                },
            )
            return result
        finally:
            connection.close()
            raw.close()
