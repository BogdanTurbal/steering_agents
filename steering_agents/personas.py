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


# Verbalized confidence field (wording of the hiring personas), added to the coding and paper reviewers' output formats
# when the run measures confidence. Peers never see it: transcripts show only score + review text.
CONFIDENCE_FIELD = '"confidence": <integer from 1 to 5, where 1 = very uncertain and 5 = very confident in your assessment>'


def with_confidence(fmt):
    """Output-format prompt with the confidence field appended to its JSON template (before the last '}')."""
    i = fmt.rindex("}")
    return f"{fmt[:i].rstrip()}, {CONFIDENCE_FIELD} {fmt[i:]}"
