"""Per-reply confidence table for a run made with `run_mas.py --confidence` (notes/confidence_plan.md).

  python scripts/confidence_table.py --run-dir experiments/<run> [--apps coding]
Writes <run>/analysis/confidence/replies_<app>.csv (one row per worker reply) and summary_<app>.json (probe health +
round-1 AUROC of each measure against the ground truth); one file per app, so the per-app jobs never write the same file.
The hypothesis tests (H1-H6) are run on the replies tables. A probe whose `written` number differs from the parsed value
is dropped (counted in probe_mismatch).

Columns: app, run, item, cond, t, agent, score, next_score, switch, peer_mean (other agents at t), think_tokens (C0),
c1_* (score distribution after thinking), c1n_* (no thinking), c2 (verbalized 1-5), c2p_* (confidence distribution),
probe_cache, y / pred / correct (coding security reviewer: code at t has no CodeQL alert, pred = score >= 3; review:
ICLR accept, pred = score >= 6), hiring: ref (mean round-1 base score of the same pair and agent over all demographic
variants) and err = |score - ref|.
"""
import argparse, glob, json, os, sys
import numpy as np
import pandas as pd
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from steering_agents.confidence import features

THRESHOLD = {"coding": 3, "hiring": 3, "review": 6}   # "deployable" / "accept" / hire-leaning: score >= threshold


def load(d):
    recs, runs = {}, {}
    for p in glob.glob(f"{d}/records_shard*.jsonl"):
        for ln in open(p):
            try: r = json.loads(ln)
            except json.JSONDecodeError: continue
            recs[r["key"]] = r
    for p in glob.glob(f"{d}/runs_shard*.jsonl"):
        for ln in open(p):
            r = json.loads(ln); runs[r["run"]] = r
    return recs, runs


def rows_for(app, d):
    recs, runs = load(d)
    thr = THRESHOLD[app]
    out = []
    for r in recs.values():
        if r["phase"] != "review" or r["parsed"].get("failed") or "score" not in r["parsed"]:
            continue
        meta = runs.get(r["run"], {})
        row = {"app": app, "run": r["run"], "item": meta.get("id", r["run"].split("#")[0]), "cond": meta.get("cond", "base"),
               "t": r["t"], "agent": r["agent"], "score": r["parsed"]["score"], "think_tokens": r["think_tokens"],
               "c2": r["parsed"].get("confidence", np.nan)}
        if app == "review":
            row["decision"] = meta.get("decision")
        if app == "hiring":
            row["pair"] = meta.get("pair")
        conf = r.get("conf") or {}
        row["probe_mismatch"] = 0
        for name, pre, th in (("score", "c1", thr), ("score_nothink", "c1n", thr), ("confidence", "c2p", None)):
            d = conf.get(name)
            if d and "written" in d and float(d["written"]) != r["parsed"].get(name):
                d, row["probe_mismatch"] = None, row["probe_mismatch"] + 1
            for k, v in features(d, th).items():
                row[f"{pre}_{k}"] = v
        row["probe_cache"] = conf.get("cache")
        out.append(row)
    df = pd.DataFrame(out)
    if df.empty:
        return df
    key = df.set_index(["run", "t", "agent"])["score"]
    df["next_score"] = [key.get((r, t + 1, a), np.nan) for r, t, a in zip(df.run, df.t, df.agent)]
    df["switch"] = (df.next_score != df.score).where(df.next_score.notna())
    tot = df.groupby(["run", "t"]).score.agg(["sum", "count"])
    s = tot.loc[list(zip(df.run, df.t))].to_numpy()
    df["peer_mean"] = (s[:, 0] - df.score) / (s[:, 1] - 1).clip(min=1)
    df["pred"] = (df.score >= thr).astype(int)
    if app == "coding":
        cq = f"{d}/codeql_results.jsonl"
        if os.path.exists(cq):
            ok = {(x["run"], x["round"]): int(not x["rules"]) for x in map(json.loads, open(cq))}
            df["y"] = [ok.get((r, t), np.nan) if a == "security" else np.nan for r, t, a in zip(df.run, df.t, df.agent)]
    elif app == "review":
        df["y"] = df.decision.astype(str).str.startswith("accept").astype(float)
    elif app == "hiring":
        base1 = df[(df.cond == "base") & (df.t == 1)].groupby(["pair", "agent"]).score.mean()
        df["ref"] = [base1.get((p, a), np.nan) for p, a in zip(df.pair, df.agent)]
        df["err"] = (df.score - df.ref).abs()
    if "y" in df:
        df["correct"] = (df.pred == df.y).astype(float).where(df.y.notna())
    return df


def auroc(score, label):
    """P(score of a positive > score of a negative), ties count 1/2 (Mann-Whitney)."""
    m = score.notna() & label.notna()
    s, y = score[m], label[m].astype(int)
    n1, n0 = int(y.sum()), int((1 - y).sum())
    if n1 == 0 or n0 == 0:
        return None
    r = s.rank()
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def summarize(df):
    """Probe health and round-1 AUROC (confidence -> correct) per measure; confident = higher value."""
    measures = {"C0 -think_tokens": -df.think_tokens, "C1 p_max": df.get("c1_p_max"), "C1 1-entropy": df.get("c1_conf_ent"),
                "C1n p_max": df.get("c1n_p_max"), "C2 verbal": df.c2, "C2p E": df.get("c2p_E")}
    s = {"replies": len(df),
         "probe_ok_share": {k: float(df[c].notna().mean()) for k, c in (("C1", "c1_mass"), ("C1n", "c1n_mass"),
                                                                        ("C2p", "c2p_mass")) if c in df},
         "mean_mass": {k: float(df[c].mean()) for k, c in (("C1", "c1_mass"), ("C1n", "c1n_mass"), ("C2p", "c2p_mass")) if c in df},
         "mean_probe_cache": float(pd.to_numeric(df.probe_cache, errors="coerce").mean()) if "probe_cache" in df else None,
         "probe_mismatch": int(df.probe_mismatch.sum())}
    if "correct" in df:
        r1 = df[df.t == 1]
        s["round1_auroc_correct"] = {k: auroc(v[r1.index], r1.correct) for k, v in measures.items() if v is not None}
        s["round1_n_labelled"] = int(r1.correct.notna().sum())
    if "err" in df:
        r1 = df[df.t == 1]
        s["round1_spearman_conf_vs_neg_err"] = {k: float(v[r1.index].rank().corr((-r1.err).rank()))   # no scipy needed
                                                for k, v in measures.items() if v is not None}
    return s


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--apps", default=",".join(THRESHOLD))
    a = ap.parse_args()
    out = f"{a.run_dir}/analysis/confidence"
    os.makedirs(out, exist_ok=True)
    clean = lambda x: {k: clean(v) for k, v in x.items()} if isinstance(x, dict) else (None if x != x else x)   # NaN -> null
    for app in a.apps.split(","):
        d = f"{a.run_dir}/{app}"
        df = rows_for(app, d) if os.path.isdir(d) else pd.DataFrame()
        if df.empty:
            continue
        df.to_csv(f"{out}/replies_{app}.csv", index=False)
        summary = clean(summarize(df))
        json.dump(summary, open(f"{out}/summary_{app}.json", "w"), indent=1)
        print(app, json.dumps(summary, indent=1))
