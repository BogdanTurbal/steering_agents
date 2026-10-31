import json
import os

_P = None


def personas():
    global _P
    if _P is None:
        path = os.path.join(os.path.dirname(__file__), "..", "data", "personas.json")
        _P = json.load(open(path))
    return _P


def persona(name):
    return personas()[name]
