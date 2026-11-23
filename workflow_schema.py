"""Validate JSON against a subset of JSON Schema, without third-party dependencies.

This is intentionally not a general JSON Schema implementation: it covers the
keywords used by the bundled framework schema and by tool ``input_schema``s.
The exported schema is also usable by a full Draft 2020-12 validator or an editor.
"""

import json
import math
import re
from functools import lru_cache
from pathlib import Path


@lru_cache(maxsize=1)
def schema_document():
    return json.loads(
        (Path(__file__).parent / "schemas" / "workflow.schema.json").read_text(encoding="utf-8")
    )


def validate_schema(value, schema, *, definitions=None, path="$") -> None:
    """Validate JSON ``value`` against ``schema``; raise ``ValueError`` naming the location.

    Supports the keywords this project uses: type, enum, properties, required,
    additionalProperties, propertyNames, items, minItems, maxItems, uniqueItems,
    minLength, maxLength, pattern, minimum, maximum, exclusiveMinimum,
    exclusiveMaximum, anyOf and local ``$ref`` (resolved in ``definitions``).
    """
    definitions = definitions or {}

    def validate(v, schema, location):
        if "$ref" in schema:
            name = schema["$ref"].split("/")[-1]
            if name not in definitions:
                raise ValueError(f"{location}: unresolvable reference {schema['$ref']!r}")
            return validate(v, definitions[name], location)
        if "anyOf" in schema:
            for choice in schema["anyOf"]:
                try:
                    validate(v, choice, location)
                    return
                except ValueError:
                    pass
            raise ValueError(f"{location}: value does not match any permitted schema")
        checks = {
            "object": isinstance(v, dict),
            "array": isinstance(v, list),
            "string": isinstance(v, str),
            "boolean": type(v) is bool,
            "integer": type(v) is int,
            "number": type(v) in (int, float) and math.isfinite(v),
            "null": v is None,
        }
        types = schema.get("type", [])
        if isinstance(types, str):
            types = [types]
        if types and not any(checks[t] for t in types):
            raise ValueError(f"{location}: expected {' or '.join(types)}")
        if "enum" in schema and not any(
            type(v) is type(option) and v == option for option in schema["enum"]
        ):
            raise ValueError(f"{location}: expected one of {schema['enum']}")
        if isinstance(v, dict):
            properties = schema.get("properties", {})
            for name in schema.get("required", []):
                if name not in v:
                    raise ValueError(f"{location}.{name}: required field missing")
            for key, item in v.items():
                if "propertyNames" in schema:
                    validate(key, schema["propertyNames"], f"{location}.{key}")
                if key in properties:
                    validate(item, properties[key], f"{location}.{key}")
                elif schema.get("additionalProperties") is False:
                    raise ValueError(
                        f"{location}.{key}: unknown field (additionalProperties=false)"
                    )
                elif isinstance(schema.get("additionalProperties"), dict):
                    validate(item, schema["additionalProperties"], f"{location}.{key}")
        if isinstance(v, list):
            if len(v) < schema.get("minItems", 0) or len(v) > schema.get("maxItems", len(v)):
                raise ValueError(f"{location}: invalid item count")
            unique = {json.dumps(x, sort_keys=True) for x in v}
            if schema.get("uniqueItems") and len(unique) != len(v):
                raise ValueError(f"{location}: duplicate items")
            for i, item in enumerate(v):
                validate(item, schema.get("items", {}), f"{location}[{i}]")
        if isinstance(v, str):
            if "pattern" in schema and not re.search(schema["pattern"], v):
                raise ValueError(f"{location}: invalid string format")
            if len(v) < schema.get("minLength", 0):
                raise ValueError(f"{location}: string is too short")
            if "maxLength" in schema and len(v) > schema["maxLength"]:
                raise ValueError(f"{location}: string is longer than {schema['maxLength']}")
        if type(v) in (int, float):
            if "minimum" in schema and v < schema["minimum"]:
                raise ValueError(f"{location}: must be >= {schema['minimum']}")
            if "exclusiveMinimum" in schema and v <= schema["exclusiveMinimum"]:
                raise ValueError(f"{location}: must be > {schema['exclusiveMinimum']}")
            if "maximum" in schema and v > schema["maximum"]:
                raise ValueError(f"{location}: must be <= {schema['maximum']}")
            if "exclusiveMaximum" in schema and v >= schema["exclusiveMaximum"]:
                raise ValueError(f"{location}: must be < {schema['exclusiveMaximum']}")

    json.dumps(value, allow_nan=False)
    validate(value, schema, path)


def validate_document(value, kind="workflow", path="$"):
    """Validate a framework document (workflow, node, tool, ...) against the bundled schema."""
    definitions = schema_document()["$defs"]
    validate_schema(value, definitions[kind], definitions=definitions, path=path)
