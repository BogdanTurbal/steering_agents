"""Recover the code produced in every round of the coding MAS from records.jsonl."""
import glob
import json
import os


def collect(run_dir):
    """(run, round) -> code, replaying the generator fallback (keep previous code if the generator failed)."""
    recs = {}
    for p in glob.glob(os.path.join(run_dir, "records_shard*.jsonl")):
        for ln in open(p):
            try:
                r = json.loads(ln)
            except json.JSONDecodeError:
                continue
            recs[r["key"]] = r
    runs = sorted({r["run"] for r in recs.values()})
    out = []
    for run in runs:
        code, rnd = None, 0
        k = f"{run}|1|init|generator"
        steps = [k] + [f"{run}|{t}|orch|generator" for t in range(1, 99)]
        for key in steps:
            if key not in recs:
                break
            o = recs[key]["parsed"].get("output")
            code = o if o else (code or "")
            rnd += 1
            out.append((run, rnd, code))
    return out
