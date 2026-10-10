"""The subset of JSON Schema (draft 2020-12) the corpus format schemas use:
type, enum, const, required, properties, additionalProperties, items,
minItems, minLength, pattern, minimum, maximum, exclusiveMinimum, and local
$ref ("#/$defs/<name>"). Course tests use only the stdlib, so this small
validator stands in for the jsonschema package."""

from __future__ import annotations

import re
from typing import Any

TYPES = {
    "object": dict,
    "array": list,
    "string": str,
    "boolean": bool,
    "null": type(None),
}


def _is(value: Any, t: str) -> bool:
    if t == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if t == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    return isinstance(value, TYPES[t])


def errors(
    value: Any, schema: dict, root: dict | None = None, at: str = "$"
) -> list[str]:
    root = root if root is not None else schema
    if "$ref" in schema:
        name = schema["$ref"].removeprefix("#/$defs/")
        return errors(value, root["$defs"][name], root, at)
    out: list[str] = []
    if "type" in schema:
        types = schema["type"] if isinstance(schema["type"], list) else [schema["type"]]
        if not any(_is(value, t) for t in types):
            return [f"{at}: {value!r} is not {schema['type']}"]
    if "enum" in schema and value not in schema["enum"]:
        out.append(f"{at}: {value!r} not in {schema['enum']}")
    if "const" in schema and value != schema["const"]:
        out.append(f"{at}: {value!r} != {schema['const']!r}")
    if isinstance(value, str):
        if len(value) < schema.get("minLength", 0):
            out.append(f"{at}: shorter than {schema['minLength']}")
        if "pattern" in schema and not re.search(schema["pattern"], value):
            out.append(f"{at}: {value!r} does not match {schema['pattern']}")
    if _is(value, "number"):
        if "minimum" in schema and value < schema["minimum"]:
            out.append(f"{at}: {value} < {schema['minimum']}")
        if "maximum" in schema and value > schema["maximum"]:
            out.append(f"{at}: {value} > {schema['maximum']}")
        if "exclusiveMinimum" in schema and value <= schema["exclusiveMinimum"]:
            out.append(f"{at}: {value} <= {schema['exclusiveMinimum']}")
    if isinstance(value, list):
        if len(value) < schema.get("minItems", 0):
            out.append(f"{at}: fewer than {schema['minItems']} items")
        if "items" in schema:
            for i, v in enumerate(value):
                out += errors(v, schema["items"], root, f"{at}[{i}]")
    if isinstance(value, dict):
        for k in schema.get("required", []):
            if k not in value:
                out.append(f"{at}: missing {k!r}")
        props = schema.get("properties", {})
        extra = schema.get("additionalProperties", True)
        for k, v in value.items():
            if k in props:
                out += errors(v, props[k], root, f"{at}.{k}")
            elif extra is False:
                out.append(f"{at}: unexpected key {k!r}")
            elif isinstance(extra, dict):
                out += errors(v, extra, root, f"{at}.{k}")
    return out
