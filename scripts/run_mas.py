"""Worker: run one application's MAS for the chosen items/trials (one shard). Resumable via records_shard*.jsonl.

  python scripts/run_mas.py --app coding --out-dir experiments/<run>/coding --n-items 20 --n-trials 3 --shard 0/1
"""
import argparse, json, os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from steering_agents.llm import Engine, DEFAULT_MODEL
from steering_agents.mas import Runner


def make_app(a):
    dk = {"data": a.data} if a.data else {}
    if a.app == "coding":
        from steering_agents.apps.coding import CodingApp
        return CodingApp(T=a.rounds, reviewers_see_all=not a.reviewers_own_only)
    if a.app == "hiring":
        from steering_agents.apps.hiring import HiringApp
        return HiringApp(T=a.rounds, adversary=a.adversary, adv_rounds=a.adv_rounds, variants=a.variants, **dk)
    from steering_agents.apps.review import ReviewApp
    return ReviewApp(T=a.rounds, **dk)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--app", required=True, choices=["coding", "hiring", "review"])
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--n-items", type=int, default=10)
    ap.add_argument("--n-trials", type=int, default=10)
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0, help="item sampling seed (same for all shards)")
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--wave", type=int, default=256, help="runs advanced together through all rounds (bigger = less idle tail per phase)")
    ap.add_argument("--reviewers-own-only", action="store_true", help="coding: reviewers see only their own past feedback (default: all critics' feedback = global state)")
    ap.add_argument("--adversary", action="store_true", help="hiring: also run the one-biased-evaluator condition")
    ap.add_argument("--adv-rounds", type=int, default=2)
    ap.add_argument("--variants", type=int, default=5, help="hiring: signal variants per demographic group (1-5)")
    ap.add_argument("--data", default=None, help="hiring: pairs.jsonl; review: papers.jsonl")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--quant", default=None, help="e.g. bitsandbytes / fp8; default: none (bf16, as resman)")
    ap.add_argument("--max-num-seqs", type=int, default=1000)
    ap.add_argument("--effort", default="medium", choices=["low", "medium", "xhigh"])
    ap.add_argument("--max-tokens", type=int, default=12288)
    ap.add_argument("--max-model-len", type=int, default=262144)
    a = ap.parse_args()
    i, n = map(int, a.shard.split("/"))
    app = make_app(a)
    done_marker = os.path.join(a.out_dir, f"done_shard{i}of{n}.json")
    if os.path.exists(done_marker):   # finished earlier (e.g. a resumed job): skip the 3-4 min model load
        print("shard already finished:", done_marker); sys.exit(0)
    runs = app.runs(a.n_items, a.n_trials, a.seed)
    print(f"{a.app}: {len(runs)} runs total, shard {i}/{n}", flush=True)
    eng = Engine(a.model, max_model_len=a.max_model_len, effort=a.effort, max_tokens=a.max_tokens, max_num_seqs=a.max_num_seqs, quant=a.quant)
    t0 = time.time()
    Runner(app, eng, a.out_dir, (i, n), a.wave, log=lambda m: print(f"[{time.time() - t0:6.0f}s] {m}", flush=True)).run(runs)
    json.dump({"args": vars(a), "seconds": time.time() - t0}, open(done_marker, "w"))
