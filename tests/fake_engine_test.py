"""CPU smoke test: every app end-to-end with a fake engine (no GPU)."""
import json, os, random, shutil, sys, tempfile
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from steering_agents.llm import Out
from steering_agents.mas import Runner
from steering_agents.apps.coding import CodingApp
from steering_agents.apps.hiring import HiringApp
from steering_agents.apps.review import ReviewApp


class Fake:
    def stream(self, batch, probes=None):
        for i, o in enumerate(self.chat(batch)):
            if probes and probes[i]:
                K, r = probes[i]["score"], random.Random(batch[i][1])
                w = [r.random() for _ in range(K)]
                o.conf = {"score": {"p": [x / sum(w) for x in w], "mass": 0.99}, "score_nothink": None,
                          "confidence": {"p": [0, 0, 0.2, 0.5, 0.3], "mass": 0.9}, "cache": 1.0}
            yield i, o

    def chat(self, batch):
        outs = []
        for msgs, seed in batch:
            r = random.Random(seed)
            u = msgs[-1]["content"]
            if "JSON object" in u and '"output"' in u and '"score"' in u:       # reviewer
                a = {"output": "looks ok", "score": r.randint(1, 10) if "1 to 10" in u else r.randint(1, 5)}
                a = json.dumps({**a, "confidence": 4} if '"confidence"' in u else a)
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
papers = os.path.join(tmp, "papers.jsonl")   # the ICLR texts are not shipped (scripts/prepare_iclr_data.py)
open(papers, "w").write("".join(json.dumps({"paper_id": f"p{k}", "title": f"T{k}", "type": "poster", "decision": d,
    "primary_area": "a", "text": "paper text"}) + "\n" for k, d in enumerate(["accept (poster)", "reject"])))
N = int(os.environ.get("N", 2))
CONF = bool(int(os.environ.get("CONF", 1)))   # also exercise the confidence path (prompt field, probes, `conf` in records)
for app in (CodingApp(T=3, confidence=CONF), HiringApp(T=3, adversary=True, variants=1, data=pairs, confidence=CONF),
            ReviewApp(T=3, data=papers, confidence=CONF)):
    runs = app.runs(N, 2, 0)
    out = os.path.join(tmp, app.name)
    Runner(app, Fake(), out, wave=4, log=lambda m: None).run(runs)
    # resume must replay with zero new generation
    class Boom:
        def stream(self, b, probes=None): raise SystemExit("resume regenerated")
    Runner(app, Boom(), out, wave=4, log=lambda m: None).run(app.runs(N, 2, 0))
    recs = [json.loads(l) for l in open(os.path.join(out, "records_shard0of1.jsonl"))]
    fails = sum(r["parsed"].get("failed", False) for r in recs)
    workers = [r for r in recs if r["phase"] == "review" and not r["parsed"].get("failed")]
    if CONF:   # every worker reply carries conf and a verbalized confidence; orchestrator replies carry neither
        assert all("conf" in r and "confidence" in r["parsed"] for r in workers), app.name
        assert not any("conf" in r for r in recs if r["phase"] != "review"), app.name
    print(app.name, "runs", len(runs), "records", len(recs), "failed-after-retries", fails)
if not os.environ.get("KEEP"): shutil.rmtree(tmp)
