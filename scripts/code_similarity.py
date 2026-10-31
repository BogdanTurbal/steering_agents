"""LLM-as-judge functional similarity of the codes produced across trials (paper B.1): for each prompt and round,
all trial pairs (45 for 10 trials) are judged Yes/No; similarity = fraction of Yes. Same model as judge (paper: cross judges).

  python scripts/code_similarity.py --run-dir experiments/<run>/coding --n-prompts 20
"""
import argparse, itertools, json, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from steering_agents.codes import collect
from steering_agents.jsonio import extract_json
from steering_agents.llm import Engine, seed_for
from steering_agents.personas import persona
from steering_agents.apps.coding import LANG, fence

ap = argparse.ArgumentParser()
ap.add_argument("--run-dir", required=True)
ap.add_argument("--n-prompts", type=int, default=20)
ap.add_argument("--effort", default="medium")
a = ap.parse_args()
out_path = os.path.join(a.run_dir, "code_similarity.jsonl")
done = {json.loads(l)["key"] for l in open(out_path)} if os.path.exists(out_path) else set()

codes = {}   # (item, round) -> {trial: code}
for run, rnd, code in collect(a.run_dir):
    item, trial = run.rsplit("#t", 1)
    codes.setdefault((item, rnd), {})[int(trial)] = code
items = sorted({i for i, _ in codes})[: a.n_prompts]
p = persona("Code Similarity Judge")
jobs = []
for (item, rnd), tc in sorted(codes.items()):
    if item not in items:
        continue
    lang = item.split("-")[0]
    for i, j in itertools.combinations(sorted(tc), 2):
        key = f"{item}|{rnd}|{i}|{j}"
        if key in done:
            continue
        u = (f"{p['Compare Prompt']}\n\n# Code A\n{fence(tc[i], lang)}\n\n# Code B\n{fence(tc[j], lang)}\n\n{p['Compare Output Format Prompt']}")
        msgs = [{"role": "system", "content": p["System Prompt"].replace("{language}", LANG[lang])}, {"role": "user", "content": u}]
        jobs.append((key, item, rnd, msgs))
print(len(jobs), "comparisons", flush=True)
if not jobs:   # e.g. 1 trial per prompt: nothing to compare, skip the model load
    sys.exit(0)
eng = Engine(effort=a.effort, max_tokens=8192)
with open(out_path, "a", buffering=1) as f:   # one line per judged pair, flushed as soon as it finishes
    for n, (i, o) in enumerate(eng.stream([(m, seed_for(k)) for k, _, _, m in jobs]), 1):
        k, item, rnd, _ = jobs[i]
        v = (extract_json(o.answer) or {}).get("verdict") if o.answer else None
        if v in ("Yes", "No"):
            f.write(json.dumps({"key": k, "item": item, "round": rnd, "same": v == "Yes"}) + "\n")
        if n % 200 == 0:
            print(f"{n}/{len(jobs)}", flush=True)
