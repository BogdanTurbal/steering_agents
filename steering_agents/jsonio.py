"""Robust extraction of the JSON object an agent was asked to return."""
import json
import re

_FENCE = re.compile(r"^```(?:json)?\s*|\s*```\s*$", re.S)


def extract_json(text):
    """First JSON object in text (handles code fences / chatter). None if there is none."""
    if not text:
        return None
    t = _FENCE.sub("", text.strip())
    dec = json.JSONDecoder()
    i = t.find("{")
    while i != -1:
        try:
            obj, _ = dec.raw_decode(t[i:])
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            pass
        i = t.find("{", i + 1)
    return None


def _num(x, lo, hi):
    try:
        v = float(x)
    except (TypeError, ValueError):
        m = re.search(r"-?\d+(\.\d+)?", str(x))
        if not m:
            return None
        v = float(m.group(0))
    return v if lo <= v <= hi else None


def parse_fields(text, str_fields=(), num_fields=(), bounds=None):
    """Parse {field: value}. Strings must be non-empty, numbers must lie in bounds[field]. None on failure.

    Fallback when the JSON is malformed (typically unescaped quotes/newlines inside a long string): regex for numbers,
    and for the single string field the text between the first `"field": "` and the last `"` before the next key.
    """
    bounds = bounds or {}
    obj = extract_json(text)
    out = {}
    if obj is not None:
        for f in str_fields:
            v = obj.get(f)
            if not isinstance(v, str) or not v.strip():
                return None
            out[f] = v
        for f in num_fields:
            v = _num(obj.get(f), *bounds.get(f, (-1e9, 1e9)))
            if v is None:
                return None
            out[f] = v
        return out
    for f in num_fields:
        m = re.search(r'"%s"\s*:\s*("?)(-?\d+(?:\.\d+)?)\1' % f, text or "")
        v = _num(m.group(2), *bounds.get(f, (-1e9, 1e9))) if m else None
        if v is None:
            return None
        out[f] = v
    for f in str_fields:
        m = re.search(r'"%s"\s*:\s*"(.*?)"\s*(?:,\s*"[a-z_]+"\s*:|\}\s*$)' % f, text or "", re.S)
        if not m or not m.group(1).strip():
            return None
        out[f] = m.group(1).replace('\\n', '\n').replace('\\"', '"')
    return out
