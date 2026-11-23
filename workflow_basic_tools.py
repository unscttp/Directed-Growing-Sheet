"""Bounded text, JSON, CSV, hashing and URL operations."""

import csv
import hashlib
import io
import json
from urllib.parse import urlencode, urlsplit, urlunsplit

from workflow_utils import get_value

# name -> (required arguments, optional arguments), each mapping to a Python type.
FILE_CONTRACTS = {
    "fs.stat": ({"path": str}, {}),
    "fs.list_directory": ({"path": str}, {}),
    "fs.read_text": ({"path": str}, {}),
    "fs.read_json": ({"path": str}, {}),
    "fs.write_text": ({"path": str, "content": str}, {"overwrite": bool}),
    "fs.make_directory": ({"path": str}, {"parents": bool}),
    "fs.delete_file": ({"path": str}, {}),
}

CONTRACTS = {
    "text.search": ({"text": str, "query": str}, {}),
    "text.replace": ({"text": str, "old": str, "new": str}, {}),
    "text.split": ({"text": str, "separator": str}, {}),
    "text.join": ({"items": list, "separator": str}, {}),
    "json.parse": ({"text": str}, {}),
    "json.stringify": ({"data": dict}, {}),
    "json.get": ({"data": dict, "pointer": str}, {}),
    "csv.parse": ({"text": str}, {}),
    "csv.format": ({"rows": list}, {}),
    "hash.sha256": ({"text": str}, {}),
    "url.build": ({"url": str, "query": dict}, {}),
    "http.request": (
        {"url": str, "method": str},
        {"headers": dict, "body": str, "query": dict, "format": str},
    ),
    "tools.define": ({"definition": dict}, {"overwrite": bool}),
}

TOOL_CONTRACTS = {**FILE_CONTRACTS, **CONTRACTS}


def strict_json(text):
    def invalid(value):
        raise ValueError(f"Invalid JSON constant: {value}")

    return json.loads(text, parse_constant=invalid)


def execute_basic(name, args, limit, max_bytes):
    if name == "text.search":
        if not args["query"]:
            raise ValueError("query must be nonempty.")
        matches = []
        offset = 0
        while len(matches) < limit:
            index = args["text"].find(args["query"], offset)
            if index < 0:
                break
            matches.append(index)
            offset = index + len(args["query"])
        return {"positions": matches, "truncated": args["text"].find(args["query"], offset) >= 0}
    if name == "text.replace":
        if not args["old"]:
            raise ValueError("old must be nonempty.")
        count = args["text"].count(args["old"])
        size = len(args["text"].encode("utf8")) + count * (
            len(args["new"].encode("utf8")) - len(args["old"].encode("utf8"))
        )
        if size > max_bytes:
            raise ValueError("Replacement exceeds max_read_bytes.")
        return {"text": args["text"].replace(args["old"], args["new"])}
    if name == "text.split":
        if not args["separator"]:
            raise ValueError("separator must be nonempty.")
        parts = args["text"].split(args["separator"], limit)
        if len(parts) > limit:
            raise ValueError("Too many split items.")
        return {"items": parts}
    if name == "text.join":
        if len(args["items"]) > limit or any(type(x) is not str for x in args["items"]):
            raise ValueError("Expected a bounded list of strings.")
        size = sum(len(x.encode("utf8")) for x in args["items"]) + max(
            0, len(args["items"]) - 1
        ) * len(args["separator"].encode("utf8"))
        if size > max_bytes:
            raise ValueError("Joined text exceeds max_read_bytes.")
        return {"text": args["separator"].join(args["items"])}
    if name == "json.parse":
        return {"data": strict_json(args["text"])}
    if name == "json.stringify":
        return {"text": json.dumps(args["data"], ensure_ascii=False, allow_nan=False)}
    if name == "json.get":
        return {"value": get_value(args["data"], args["pointer"])}
    if name == "csv.parse":
        rows = []
        for row in csv.reader(io.StringIO(args["text"], newline=""), strict=True):
            if len(rows) >= limit or len(row) > limit:
                raise ValueError("CSV exceeds row or column limit.")
            rows.append(row)
        return {"rows": rows}
    if name == "csv.format":
        if len(args["rows"]) > limit or any(
            not isinstance(row, list)
            or len(row) > limit
            or any(type(cell) is not str for cell in row)
            for row in args["rows"]
        ):
            raise ValueError("Expected bounded rows of strings.")
        stream = io.StringIO(newline="")
        csv.writer(stream).writerows(args["rows"])
        return {"text": stream.getvalue()}
    if name == "hash.sha256":
        return {"sha256": hashlib.sha256(args["text"].encode("utf-8")).hexdigest()}
    if name == "url.build":
        parts = urlsplit(args["url"])
        if parts.scheme not in {"http", "https"} or not parts.hostname:
            raise ValueError("Expected an HTTP(S) URL.")
        if any(
            type(k) is not str or type(v) not in (str, int, bool) for k, v in args["query"].items()
        ):
            raise ValueError("Query values must be scalar strings, integers or booleans.")
        query = "&".join(x for x in (parts.query, urlencode(args["query"])) if x)
        return {"url": urlunsplit(parts._replace(query=query))}
    raise KeyError(name)
