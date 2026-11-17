"""Does adding the confidence field change the system? Compare a run without --confidence (A, e.g. the medium run of
2026-10-26) with a --confidence run (B) on the same items (notes/confidence_plan.md §6, "prompt-effect check").

  python scripts/compare_runs.py --a experiments/<medium_without_conf> --b experiments/<medium_with_conf>
Writes <B>/analysis/confidence/prompt_effect.json. Per app (main conditions only, runs matched by id: same item, trial,
condition and, through seed_for, the same seeds; the replies still differ because the prompts differ):
- paired score difference B - A per round (mean and a 95% bootstrap CI over items; hiring: over job/resume pairs),
- conformity / polarization / inertia curves of both runs on the matched runs,
- parse failures, retries and thinking length of the worker replies of the matched runs in both runs,
- score histograms of round 1.
No difference beyond sampling noise = the confidence field can be treated as inert for the homogenization results.
"""
import argparse, json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from analyze import AGENTS, SCALE, load, eff_T, score_tensor   # noqa: E402
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from steering_agents import metrics as M   # noqa: E402


def boot_ci(x, n=2000, seed=0):
    """95% percentile CI of the mean over items (x: one value per item)."""
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    if len(x) < 2:
        return [float("nan")] * 2
    r = np.random.default_rng(seed)
    m = r.choice(x, (n, len(x))).mean(1)
    return [float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]


def worker_health(recs, ids):
    w = [r for r in recs.values() if r["phase"] == "review" and r["run"] in ids]
    if not w:
        return {}
    return {"replies": len(w), "failed_%": 100 * float(np.mean([bool(r["parsed"].get("failed")) for r in w])),
            "retried_%": 100 * float(np.mean([r["attempts"] > 1 for r in w])),
            "think_tokens_mean": float(np.mean([r["think_tokens"] for r in w]))}


def compare(app, da, db):
    ra, runs_a = load(f"{da}/{app}")
    rb, runs_b = load(f"{db}/{app}")
    common = {k: v for k, v in runs_a.items() if k in runs_b}
    if not common:
        return None
    T = min(eff_T(app, ra, common), eff_T(app, rb, common))
    if not T:
        return None
    Sa, Sb = score_tensor(app, ra, common, T), score_tensor(app, rb, common, T)
    ids = sorted(set(Sa) & set(Sb))
    if not ids:
        return None
    A, B = np.stack([Sa[i] for i in ids]), np.stack([Sb[i] for i in ids])   # [R, T, agents]
    lo, hi = SCALE[app]
    # bootstrap cluster: the item, except in hiring, where the demographic variants of one job/resume pair are not
    # independent, so the cluster is the pair
    item = [common[i].get("pair" if app == "hiring" else "id", i.split("#")[0]) for i in ids]
    d = np.nanmean(B - A, axis=2)   # per run and round: mean over agents of the paired difference
    per_item = {}
    for it, row in zip(item, d):
        per_item.setdefault(it, []).append(row)
    di = np.array([np.nanmean(v, axis=0) for v in per_item.values()])   # [items, T]
    out = {"matched_runs": len(ids), "clusters": len(per_item), "rounds": T,
           "score_diff_B_minus_A": {"mean": np.nanmean(di, 0).tolist(), "ci95": [boot_ci(di[:, t]) for t in range(T)]}}
    for name, X in (("A", A), ("B", B)):
        out[name] = {"conformity": M.conformity_curve(X).tolist(), "polarization": M.polarization_curve(X, lo, hi).tolist(),
                     "inertia": M.inertia_curve(X).tolist(), "definition_rates": M.definition_rates(X, lo, hi),
                     "round1_hist": {int(v): int((X[:, 0] == v).sum()) for v in np.unique(X[:, 0][~np.isnan(X[:, 0])])}}
    out["A"]["health"], out["B"]["health"] = worker_health(ra, set(ids)), worker_health(rb, set(ids))
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True, help="run dir without --confidence")
    ap.add_argument("--b", required=True, help="run dir with --confidence")
    a = ap.parse_args()
    res = {app: compare(app, a.a, a.b) for app in AGENTS if os.path.isdir(f"{a.a}/{app}") and os.path.isdir(f"{a.b}/{app}")}
    os.makedirs(f"{a.b}/analysis/confidence", exist_ok=True)
    json.dump(res, open(f"{a.b}/analysis/confidence/prompt_effect.json", "w"), indent=1)
    for app, r in res.items():
        if r:
            print(app, "matched runs", r["matched_runs"], "score diff by round", np.round(r["score_diff_B_minus_A"]["mean"], 2),
                  "conformity A", np.round(r["A"]["conformity"], 2), "B", np.round(r["B"]["conformity"], 2))
