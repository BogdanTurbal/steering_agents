"""Local smoke test with a real model and no GPU: the real Engine, Runner and apps with confidence probes, on a small Qwen
served by the transformers stand-in for vLLM (tests/local_vllm). Checks the real tokenizer, chat template, parsing and
probe positions; it cannot check vLLM itself (enqueue of token prompts, prefix cache, speed).

  pip install torch transformers accelerate numpy pandas            # Python >= 3.10, no vllm
  PYTHONPATH=tests/local_vllm python tests/local_model_smoke.py --out /tmp/lms
  python scripts/confidence_table.py --run-dir /tmp/lms/run          # probe health per app

Default: Qwen/Qwen3-0.6B (downloaded on first use), 2 rounds, 2 hiring runs, 2 coding stubs, 1 short made-up paper;
~10 min on an Apple-silicon laptop.
"""
import argparse, json, os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import vllm
assert "local_vllm" in vllm.__file__, "put tests/local_vllm first on PYTHONPATH (this test must not use a real vLLM)"
from huggingface_hub import snapshot_download
from steering_agents.llm import Engine
from steering_agents.mas import Runner
from steering_agents.apps.coding import CodingApp
from steering_agents.apps.hiring import HiringApp
from steering_agents.apps.review import ReviewApp

PAPER = {"paper_id": "p0", "title": "Sparse Mixture of Adapters for Continual Learning", "type": "poster",
         "decision": "accept (poster)", "primary_area": "transfer learning",
         "text": "Abstract. We propose sparse mixtures of low-rank adapters that are routed per task, so that new tasks do "
                 "not overwrite old ones. On Split-CIFAR100 and a 20-task NLP benchmark the method reduces forgetting by "
                 "31% relative to LoRA fine-tuning at equal parameter count, with ablations on routing sparsity and rank. "
                 "Limitations: tasks must be known at test time; compute grows with the number of tasks."}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--apps", default="hiring,coding,review")
    ap.add_argument("--rounds", type=int, default=2)
    a = ap.parse_args()
    eng = Engine(snapshot_download(a.model), max_model_len=40960, max_tokens=2500)
    os.makedirs(a.out, exist_ok=True)
    paper = os.path.join(a.out, "papers.jsonl")
    open(paper, "w").write(json.dumps(PAPER) + "\n")
    apps = {"hiring": (HiringApp(T=a.rounds, adversary=True, variants=1, confidence=True), 1, 2),   # base + adv, 1 group
            "coding": (CodingApp(T=a.rounds, confidence=True), 2, 2),
            "review": (ReviewApp(T=a.rounds, data=paper, confidence=True), 1, 1)}
    for name in a.apps.split(","):
        app, n_items, n_runs = apps[name]
        t0 = time.time()
        runs = app.runs(n_items, 1, 0)[:n_runs]
        Runner(app, eng, os.path.join(a.out, "run", name), wave=8, log=lambda m: print(name, m, flush=True)).run(runs)
        print(name, "done", round(time.time() - t0), "s", eng.llm.stats, flush=True)
