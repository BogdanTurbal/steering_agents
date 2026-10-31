"""CPU smoke test: every app end-to-end with a fake engine (no GPU)."""
import json, os, random, shutil, sys, tempfile
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from steering_agents.llm import Out
from steering_agents.mas import Runner
from steering_agents.apps.coding import CodingApp
from steering_agents.apps.hiring import HiringApp
from steering_agents.apps.review import ReviewApp


class Fake:
    def stream(self, batch):
        for i, o in enumerate(self.chat(batch)):
            yield i, o

    def chat(self, batch):
        outs = []
        for msgs, seed in batch:
            r = random.Random(seed)
            u = msgs[-1]["content"]
            if "JSON object" in u and '"output"' in u and '"score"' in u:       # reviewer
                a = json.dumps({"output": "looks ok", "score": r.randint(1, 10) if "1 to 10" in u else r.randint(1, 5)})
            elif '"score"' in u and '"reasoning"' in u:                         # hiring
                a = json.dumps({"score": r.randint(1, 5), "reasoning": "fine", "confidence": 3})
            elif '"decision"' in u:                                              # meta
                a = json.dumps({"output": "summary", "decision": 3})
            elif "full code" in u or "full revised code" in u:                   # generator
                a = json.dumps({"output": "def f():\n    return 1\n"})
            else:
                a = json.dumps({"output": "rebuttal"})
            if r.random() < 0.1:
                a = "garbage"                                                    # exercise the retry path
            outs.append(Out(a, 5, 20, "stop"))
        return outs


tmp = os.environ.get("KEEP") or tempfile.mkdtemp()
os.makedirs(tmp, exist_ok=True)
pairs = os.path.join(tmp, "pairs.jsonl")
open(pairs, "w").write(json.dumps({"pair_id": "00-0", "occupation": "Cooks", "jd": "JD",
    "resume": "R\nLanguages: [LANGUAGES]\nActivities: [ACTIVITIES]\nVolunteering: [VOLUNTEERING]\nHobbies: [HOBBIES]"}) + "\n")
N = int(os.environ.get("N", 2))
for app in (CodingApp(T=3), HiringApp(T=3, adversary=True, variants=1, data=pairs), ReviewApp(T=3)):
    runs = app.runs(N, 2, 0)
    out = os.path.join(tmp, app.name)
    Runner(app, Fake(), out, wave=4, log=lambda m: None).run(runs)
    # resume must replay with zero new generation
    class Boom:
        def stream(self, b): raise SystemExit("resume regenerated")
    Runner(app, Boom(), out, wave=4, log=lambda m: None).run(app.runs(N, 2, 0))
    recs = [json.loads(l) for l in open(os.path.join(out, "records_shard0of1.jsonl"))]
    fails = sum(r["parsed"].get("failed", False) for r in recs)
    print(app.name, "runs", len(runs), "records", len(recs), "failed-after-retries", fails)
if not os.environ.get("KEEP"): shutil.rmtree(tmp)
