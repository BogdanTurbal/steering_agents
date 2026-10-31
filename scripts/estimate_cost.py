"""GPU-time estimate per application/tier from the smoke-run measurements (experiments/smoke3_*; 1 x H200, Qwen3.8-27B bnb-4bit,
medium effort). NOT a benchmark: see ASSUMPTIONS. Recalibrate by editing the constants once a real run reports
"gen tok/s" (logged by the Runner and in analysis/quality_report.md).

  python scripts/estimate_cost.py                       # the launcher tiers
  python scripts/estimate_cost.py --coding 280x10 --hiring 40x10x1 --review 300x10 --target-hours 8
"""
import argparse, math

# --- measured in smoke3 (2 rounds, 4 coding prompts, 2 papers) -------------------------------------------------
GEN_TOK = {   # mean generated tokens (thinking + answer) per call
    "coding": {"init": 650, "review": 1000, "orch": 1700},          # measured
    "review": {"review": 4900, "orch": 4000},                        # reviewers measured (12 calls); meta/author ASSUMED
    "hiring": {"review": 1400},                                      # ASSUMED (no hiring replies were collected): like coding reviewers + 4-5 sentence reasoning
}
# per-sequence decode speed measured at 8 and 32 concurrent sequences: 15.9 and 11.5 tok/s.
# Fit r(c) = R0 / (1 + c/C0)  ->  R0 = 18.2 tok/s, C0 = 55  =>  GPU throughput(c) = c * r(c)   (EXTRAPOLATED beyond 32)
R0, C0 = 18.2, 55.0
CONC = {"coding": 96, "hiring": 96, "review": 40}   # max_num_seqs=96; review is KV-limited: ~1.58M KV tokens / ~40k-token contexts
TAIL = 0.8          # efficiency loss from phase barriers (every phase waits for its slowest call)
PREFILL_TOK_S = 5000   # ASSUMED uncached prefill speed (27B, ~54 GFLOP/token on one H200)
UNCACHED_PROMPT = {"coding": 4000, "hiring": 3000, "review": 12000}   # ASSUMED uncached prompt tokens per call (prefix caching on)

# --- structure of one run ------------------------------------------------------------------------------------------
def calls_per_run(app, T):
    return {"coding": {"init": 1, "review": 4 * T, "orch": T}, "hiring": {"review": 4 * T}, "review": {"review": 3 * T, "orch": 2 * T}}[app]


def estimate(app, runs, T=5, shards=1):
    cp = calls_per_run(app, T)
    calls = {k: v * runs for k, v in cp.items()}
    gen = sum(calls[k] * GEN_TOK[app][k] for k in calls)
    c = CONC[app]
    thr = c * R0 / (1 + c / C0) * TAIL
    decode_h = gen / thr / 3600
    prefill_h = sum(calls.values()) * UNCACHED_PROMPT[app] / PREFILL_TOK_S / 3600
    gpu_h = decode_h + prefill_h
    return {"runs": runs, "calls": sum(calls.values()), "gen_Mtok": gen / 1e6, "tok_s_per_gpu": thr, "decode_h": decode_h,
            "prefill_h": prefill_h, "gpu_h": gpu_h, "wall_h": gpu_h / shards, "shards": shards}


def parse(spec, n):
    return [int(x) for x in spec.split("x")] if spec else None


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--coding"); ap.add_argument("--hiring"); ap.add_argument("--review")
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--target-hours", type=float, default=0, help="also print the shard count that gives this wall time per job")
    a = ap.parse_args()
    if a.coding or a.hiring or a.review:
        tiers = {"custom": {"coding": parse(a.coding, 2), "hiring": parse(a.hiring, 3), "review": parse(a.review, 2)}}
    else:   # same sizes as scripts/launch_mas_pipeline.py TIERS: coding items x trials; hiring pairs x trials x variants; review papers x trials
        tiers = {"smoke(2 rounds)": {"coding": [4, 1], "hiring": [1, 1, 1], "review": [2, 1]},
                 "pilot": {"coding": [40, 1], "hiring": [10, 1, 1], "review": [60, 1]},
                 "medium": {"coding": [280, 1], "hiring": [40, 1, 1], "review": [300, 1]},
                 "full": {"coding": [280, 1], "hiring": [195, 1, 5], "review": [1919, 1]}}
    print(f"{'tier':<16}{'app':<8}{'runs':>8}{'calls':>10}{'gen Mtok':>10}{'GPU-h':>9}  (decode+prefill)  shards->wall h")
    for name, t in tiers.items():
        T = 2 if name.startswith("smoke") else a.rounds
        tot = 0
        for app in ("coding", "hiring", "review"):
            if not t.get(app):
                continue
            if app == "hiring":
                pairs, trials, variants = t[app]; runs = pairs * 8 * variants * trials * 2   # x2: base + adversary condition
            else:
                runs = t[app][0] * t[app][1]
            e = estimate(app, runs, T)
            tot += e["gpu_h"]
            sh = math.ceil(e["gpu_h"] / a.target_hours) if a.target_hours else None
            print(f"{name:<16}{app:<8}{e['runs']:>8}{e['calls']:>10}{e['gen_Mtok']:>10.1f}{e['gpu_h']:>9.1f}  ({e['decode_h']:.1f}+{e['prefill_h']:.1f})"
                  + (f"  {sh} shards -> {e['gpu_h'] / sh:.1f} h each" if sh else ""))
        print(f"{name:<16}{'TOTAL':<8}{'':>28}{tot:>9.1f}")
