"""Confidence of an agent's reply, read from the model's next-token distribution (notes/confidence_plan.md, C1/C1n/C2p).

A probe is a 1-token request on a token prefix that ends right before a number the agent wrote (its score, its verbalized
confidence). Its top logprobs give p(k) over the scale values. Probes are sent by Engine.stream as soon as the main reply
finishes, so the prefix (prompt + thinking + answer up to the number) is still in vLLM's prefix cache.

Per reply, `conf` = {"score": {"p": [p(1)..p(K)], "mass": m} | None,     C1: after thinking, at the written score
                     "score_nothink": {...} | None,                      C1n: thinking disabled, forced `{"score": `
                     "confidence": {...} | None,                         C2p: at the written 1-5 confidence
                     "cache": fraction of probe prompt tokens served from the prefix cache}
`mass` = probability on the valid digits before renormalizing (low mass = the probe was not at a number); `written` = the
number the agent actually wrote at the probe position (scripts/confidence_table.py drops a probe whose `written` differs
from the parsed value, e.g. when the regex hit a key-like string inside malformed JSON).
C0 (thinking length) and C2 (the verbalized number) are already in the record (`think_tokens`, `parsed["confidence"]`).
"""
import math
import re

TOP_LOGPROBS = 20          # vLLM's default max_logprobs
NOTHINK_PREFIX = '{"score": '


def digit_ids(tok):
    """Token id of each single digit "0".."9" (Qwen tokenizes numbers digit by digit)."""
    ids = {}
    for d in "0123456789":
        enc = tok.encode(d, add_special_tokens=False)
        if len(enc) != 1:
            raise ValueError(f"digit {d!r} is not a single token: {enc}")
        ids[d] = enc[0]
    return ids


def splits_ten(tok):
    """True if "10" is written as the two tokens "1","0" in a JSON value (needed for the 1/10 split of K=10 scales)."""
    d = digit_ids(tok)
    enc = tok.encode('{"score": 10}', add_special_tokens=False)
    return any(enc[k:k + 2] == [d["1"], d["0"]] for k in range(len(enc) - 1))


def locate(tok, out_ids, field):
    """(j, written): out_ids[:j] ends right before the first digit of `"field": <number>` in the answer (after </think>),
    written = that number as text; None if not found. Character offsets are mapped to token indices by binary search on
    the decoded prefix length."""
    text = tok.decode(out_ids)
    start = text.rfind("</think>")
    if start < 0:   # thinking never closed: no answer to probe
        return None
    m = re.compile(r'"%s"\s*:\s*"?(?=\d)' % re.escape(field)).search(text, start)
    if not m:
        return None
    off = m.end()
    lo, hi = 0, len(out_ids)
    while lo < hi:   # smallest j with len(decode(out_ids[:j])) >= off
        mid = (lo + hi) // 2
        if len(tok.decode(out_ids[:mid])) < off:
            lo = mid + 1
        else:
            hi = mid
    first = tok.decode(out_ids[lo:lo + 1]) if lo < len(out_ids) else ""
    if tok.decode(out_ids[:lo]) != text[:off] or len(first) != 1 or not first.isdigit():
        return None   # the number does not start at a token boundary, or its first token is not a single digit
    return lo, re.match(r"\d+", text[off:]).group(0)


def probe_prefixes(tok, prompt_ids, out_ids, spec, nothink_ids=None, ten_ok=True):
    """spec: {field: K} for the fields to probe (score on a 1..K scale, confidence on 1..5). Returns
    ({probe name: token prefix}, {field: written number}). For K >= 10 a second probe after the digit "1" splits p("1")
    into 1 vs 10; if the tokenizer does not write "10" as "1","0" (ten_ok=False), K >= 10 fields are not probed."""
    one = digit_ids(tok)["1"]
    pre, written = {}, {}
    for field, K in spec.items():
        if K >= 10 and not ten_ok:
            continue
        loc = locate(tok, out_ids, field)
        if loc is not None:
            pre[field] = prompt_ids + out_ids[:loc[0]]
            written[field] = loc[1]
        if field == "score" and nothink_ids is not None:
            pre["score_nothink"] = nothink_ids
    for name in list(pre):
        if spec[name.split("_")[0]] >= 10:
            pre[name + "|1"] = pre[name] + [one]
    return pre, written


def distribution(top, K, digits, top_after_one=None):
    """top: {token_id: logprob} at the probe position. p over 1..K (renormalized) and the raw mass on valid values.
    Digits outside the top logprobs count as 0."""
    pr = {d: math.exp(top[i]) if i in top else 0.0 for d, i in digits.items()}
    p = [pr[str(k)] for k in range(1, min(K, 9) + 1)]
    if K >= 10:
        q = 0.0   # P("0" | "1"): the share of p("1") that is the start of "10"
        if top_after_one:
            q = math.exp(top_after_one[digits["0"]]) if digits["0"] in top_after_one else 0.0
        p[0] = pr["1"] * (1 - q)
        p.append(pr["1"] * q)
    mass = sum(p)
    if mass <= 0:
        return None
    return {"p": [round(x / mass, 5) for x in p], "mass": round(mass, 5)}


def assemble(spec, tops, digits, cache=None, written=None):
    """tops: {probe name: {token_id: logprob}} from probe_prefixes' probes -> the `conf` dict of a record."""
    conf = {}
    for name in ("score", "score_nothink", "confidence"):
        field = name.split("_")[0]
        if field not in spec or name not in tops:
            conf[name] = None
            continue
        conf[name] = distribution(tops[name], spec[field], digits, tops.get(name + "|1"))
        if conf[name] is not None and name in (written or {}):
            conf[name]["written"] = written[name]
    conf["cache"] = cache
    return conf


def features(d, threshold=None):
    """Scalar confidence features of a distribution d = {"p": [...], ...} over 1..K (None if d is None).
    p_max, conf_ent = 1 - normalized entropy, E = expected value, sd, p_hi = P(value >= threshold)."""
    if not d:
        return {}
    p = d["p"]
    K = len(p)
    E = sum((k + 1) * x for k, x in enumerate(p))
    H = -sum(x * math.log(x) for x in p if x > 0) / math.log(K)
    f = {"p_max": max(p), "conf_ent": 1 - H, "E": E,
         "sd": math.sqrt(sum((k + 1 - E) ** 2 * x for k, x in enumerate(p))), "mass": d["mass"]}
    if threshold is not None:
        f["p_hi"] = sum(p[threshold - 1:])
    return f
