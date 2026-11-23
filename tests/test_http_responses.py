"""http.request responses: charsets, compression, size limit, text format, blocks."""

import gzip
import socket
import unittest
import zlib
from types import SimpleNamespace
from unittest.mock import patch

from workflow_http_tools import BLOCK_NOTE, HTTPPolicy, html_to_text

ADDRESSES = [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", 443))]
PAGE = (
    "<!doctype html><html><head><meta charset='gbk'><title>年鉴学派 - 维基</title>"
    "<style>body{color:red}</style><script>var x = 1;</script></head>"
    "<body><nav>首页</nav><h1>年鉴学派</h1><p>年鉴学派是二十世纪<b>法国</b>史学流派。</p>"
    "<ul><li>布洛赫</li><li>费弗尔</li></ul><noscript>请启用脚本</noscript></body></html>"
)


class ResponseTests(unittest.TestCase):
    def setUp(self):
        self.policy = HTTPPolicy(
            ["https://site.example"], {"https://site.example": {"X-Api-Key": "secret"}}
        )
        self.events = []
        self.context = SimpleNamespace(
            record_activity=lambda kind, details: self.events.append(details)
        )

    def fetch(self, body: bytes, response_headers, *, status=200, max_bytes=262144, **args):
        with (
            patch("workflow_http_tools.socket.getaddrinfo", return_value=ADDRESSES),
            patch("workflow_http_tools.socket.socket"),
            patch("workflow_http_tools.ssl.create_default_context"),
            patch("workflow_http_tools.http.client.HTTPConnection") as connection,
        ):
            response = connection.return_value.getresponse.return_value
            response.status = status
            response.getheaders.return_value = response_headers
            response.read.side_effect = lambda size: body[:size]
            result = self.policy.request(
                self.context,
                {"url": "https://site.example/page", "method": "GET", **args},
                max_bytes,
            )
            self.sent = connection.return_value.request.call_args.kwargs["headers"]
        return result

    def test_declared_and_meta_charsets(self):
        result = self.fetch(PAGE.encode("gbk"), [("Content-Type", "text/html; charset=GBK")])
        self.assertEqual(result["encoding"], "gbk")
        self.assertIn("年鉴学派是二十世纪", result["body"])
        result = self.fetch(PAGE.encode("gbk"), [("Content-Type", "text/html")])
        self.assertEqual(result["encoding"], "gbk")  # from <meta charset>
        latin = self.fetch(
            "café".encode("latin-1"), [("Content-Type", "text/plain; charset=iso-8859-1")]
        )
        self.assertEqual(latin["body"], "café")
        unknown = self.fetch(b"plain", [("Content-Type", "text/plain; charset=nonsense")])
        self.assertEqual((unknown["encoding"], unknown["body"]), ("utf-8", "plain"))

    def test_text_format_returns_title_and_visible_text(self):
        result = self.fetch(PAGE.encode("gbk"), [("Content-Type", "text/html")], format="text")
        self.assertEqual(result["title"], "年鉴学派 - 维基")
        self.assertEqual(
            result["body"], "首页\n年鉴学派\n年鉴学派是二十世纪法国史学流派。\n布洛赫\n费弗尔"
        )
        self.assertNotIn("var x", result["body"])
        json_body = self.fetch(b'{"a": 1}', [("Content-Type", "application/json")], format="text")
        self.assertEqual(json_body["body"], '{"a": 1}')  # not HTML: unchanged
        self.assertNotIn("title", json_body)

    def test_compressed_bodies_are_decoded_within_the_limit(self):
        for encoding, data in (
            ("gzip", gzip.compress(b"hello")),
            ("deflate", zlib.compress(b"hello")),
        ):
            with self.subTest(encoding=encoding):
                result = self.fetch(data, [("Content-Encoding", encoding)])
                self.assertEqual(result["body"], "hello")
        bomb = gzip.compress(b"a" * 5_000_000)
        result = self.fetch(bomb, [("Content-Encoding", "gzip")], max_bytes=1000)
        self.assertEqual((len(result["body"]), result["truncated"]), (1000, True))
        with self.assertRaises(ValueError):
            self.fetch(b"x", [("Content-Encoding", "br")])

    def test_large_bodies_are_truncated_not_refused(self):
        result = self.fetch(b"x" * 5000, [("Content-Type", "text/plain")], max_bytes=1000)
        self.assertEqual((len(result["body"]), result["truncated"]), (1000, True))
        self.assertTrue(self.events[-1]["truncated"])
        self.assertFalse(self.fetch(b"short", [])["truncated"])

    def test_default_headers_and_their_limits(self):
        self.fetch(b"ok", [])
        self.assertIn("directed-growing-sheet", self.sent["User-Agent"])
        self.assertEqual(self.sent["Accept-Encoding"], "identity")
        self.assertEqual(self.sent["X-Api-Key"], "secret")
        self.fetch(b"ok", [], headers={"user-agent": "custom/1.0", "Accept": "application/json"})
        self.assertEqual(self.sent["user-agent"], "custom/1.0")
        self.assertNotIn("User-Agent", self.sent)
        self.assertEqual(self.sent["Accept"], "application/json")
        with self.assertRaises(ValueError):
            self.fetch(b"ok", [], headers={"Accept-Encoding": "gzip"})
        with self.assertRaises(ValueError):
            self.fetch(b"ok", [], format="pdf")

    def test_blocks_and_challenges_are_flagged(self):
        self.assertEqual(self.fetch(b"denied", [], status=403)["note"], BLOCK_NOTE)
        challenge = b"<html><title>Just a moment...</title><body>cf-chl</body></html>"
        self.assertEqual(self.fetch(challenge, [("Content-Type", "text/html")])["note"], BLOCK_NOTE)
        self.assertNotIn("note", self.fetch(b"<p>fine</p>", [("Content-Type", "text/html")]))

    def test_html_to_text_handles_broken_markup(self):
        title, text = html_to_text("<title>T</title><p>one<p>two &amp; three<br>four<script>x")
        self.assertEqual((title, text), ("T", "one\ntwo & three\nfour"))


if __name__ == "__main__":
    unittest.main()
