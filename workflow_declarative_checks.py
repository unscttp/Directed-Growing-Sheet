"""Static analysis for the declarative JSON syntax tree."""

import re

# Stable result fields. None means the value's type depends on runtime data.
RESULT_FIELDS = {
    "fs.stat": {
        "path": str,
        "exists": bool,
        "is_file": bool,
        "is_directory": bool,
        "size_bytes": None,
    },
    "fs.list_directory": {"path": str, "entries": list, "truncated": bool},
    "fs.read_text": {"path": str, "content": str},
    "fs.read_json": {"path": str, "data": None},
    "fs.write_text": {"path": str, "bytes_written": int},
    "fs.make_directory": {"path": str, "is_directory": bool},
    "fs.delete_file": {"path": str, "deleted": bool},
    "text.search": {"positions": list, "truncated": bool},
    "text.replace": {"text": str},
    "text.split": {"items": list},
    "text.join": {"text": str},
    "json.parse": {"data": None},
    "json.stringify": {"text": str},
    "json.get": {"value": None},
    "csv.parse": {"rows": list},
    "csv.format": {"text": str},
    "hash.sha256": {"sha256": str},
    "url.build": {"url": str},
    "http.request": {
        "url": str,
        "status": int,
        "headers": list,
        "content_type": str,
        "encoding": str,
        "truncated": bool,
        "body": str,
    },
}


def check_tree_bounds(value):
    """Limit analysis work before recursive schema validation or tree traversal."""
    pending = [(value, 0)]
    count = 0
    while pending:
        node, depth = pending.pop()
        count += 1
        if depth > 32 or count > 10000:
            raise ValueError("Template AST exceeds depth or node limit.")
        if isinstance(node, dict):
            if any(type(key) is not str for key in node):
                raise ValueError("Template object keys must be strings.")
            pending.extend((item, depth + 1) for item in node.values())
        elif isinstance(node, list):
            pending.extend((item, depth + 1) for item in node)


def expression_type(value, parameters, previous):
    """Validate every AST node and infer types known before execution."""
    if isinstance(value, list):
        for item in value:
            expression_type(item, parameters, previous)
        return list
    if not isinstance(value, dict):
        return type(value)
    if "$input" in value:
        name = value["$input"]
        if set(value) != {"$input"} or type(name) is not str or name not in parameters:
            raise ValueError("Unknown input reference.")
        return parameters[name]
    if "$step" in value:
        name = value["$step"]
        pointer = value.get("path", "")
        if (
            set(value) - {"$step", "path"}
            or type(name) is not str
            or name not in previous
            or type(pointer) is not str
            or pointer
            and not pointer.startswith("/")
            or re.search(r"~(?![01])", pointer)
        ):
            raise ValueError(
                "Step reference must target an earlier step with a valid JSON Pointer."
            )
        if not pointer:
            return dict
        fields = RESULT_FIELDS[previous[name]]
        parts = pointer[1:].split("/")
        field = parts[0].replace("~1", "/").replace("~0", "~")
        if field not in fields:
            raise ValueError("Unknown result field in step reference.")
        result_type = fields[field]
        if len(parts) > 1:
            if result_type in (str, int, bool):
                raise ValueError("JSON Pointer traverses a scalar result.")
            if result_type is list and not re.fullmatch(r"0|[1-9][0-9]*", parts[1]):
                raise ValueError("List result needs a nonnegative index.")
            return None
        return result_type
    if any(key.startswith("$") for key in value):
        raise ValueError("Unknown template AST operator.")
    for item in value.values():
        expression_type(item, parameters, previous)
    return dict
