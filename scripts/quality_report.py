"""Automatic quality report for a run dir (<run>/{coding,hiring,review}): is the pipeline healthy and are the agents sane?
Writes <run>/analysis/quality_report.md and quality_report.json. Works on partial results.

  python scripts/quality_report.py --run-dir experiments/<run>

Sections: (1) per-agent call health: failed-after-retries, retried, truncated, prompt-too-long, tokens, throughput;
(2) score sanity: distribution per agent and round, endpoint share, agent-vs-agent disagreement;
(3) coding: code validity (Python ast.parse, C++ g++ -fsyntax-only sample), code growth, CodeQL summary;
(4) hiring: adversary vs other agents; (5) pointers to the homogenization metrics in summary.json.
"""
import argparse, glob, json, os, subprocess, sys, tempfile, ast
from collections import defaultdict
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from analyze import AGENTS, SCALE, load, eff_T, rounds_done, score_tensor   # noqa: E402
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from steering_agents.codes import collect   # noqa: E402


def pct(x, q):
    return float(np.percentile(x, q)) if len(x) else float("nan")


def health(recs):
    by = defaultdict(list)
    for r in recs.values():
        by[(r["phase"], r["agent"])].append(r)
    rows = []
    for (ph, ag), v in sorted(by.items()):
        n = len(v)
        rows.append({"phase": ph, "agent": ag, "calls": n,
                     "failed_%": 100 * sum(bool(r["parsed"].get("failed")) for r in v) / n,
                     "retried_%": 100 * sum(r["attempts"] > 1 for r in v) / n,
                     "truncated_%": 100 * sum(r["finish"] == "length" for r in v) / n,
                     "prompt_too_long": sum(r["finish"] == "prompt_too_long" for r in v),
                     "think_tok_mean": float(np.mean([r["think_tokens"] for r in v])),
                     "gen_tok_mean": float(np.mean([r["n_tokens"] for r in v])),
                     "gen_tok_p95": pct([r["n_tokens"] for r in v], 95)})
    ts = [r["time"] for r in recs.values() if "time" in r]
    toks = sum(r["n_tokens"] for r in recs.values())
    span = (max(ts) - min(ts)) if len(ts) > 1 else float("nan")
    return rows, {"calls": len(recs), "gen_tokens": toks, "wall_s_between_first_last_reply": span,
                  "gen_tok_per_s_overall": toks / span if span == span and span > 0 else float("nan")}


def score_sanity(app, recs, runs):
    T = eff_T(app, recs, runs)
    if not T:
        return None
    S = score_tensor(app, recs, runs, T)
    if not S:
        return None
    A = np.stack(list(S.values()))   # [R, T, agents]
    lo, hi = SCALE[app]
    out = {"runs": len(S), "rounds": T, "per_agent": {}, "disagreement_by_round": [], "endpoint_share_by_round": []}
    for j, ag in enumerate(AGENTS[app]):
        x = A[:, :, j]
        out["per_agent"][ag] = {"mean_by_round": np.nanmean(x, 0).round(2).tolist(), "std_by_round": np.nanstd(x, 0).round(2).tolist(),
                                "hist": {int(v): int((x == v).sum()) for v in np.unique(x[~np.isnan(x)])}}
    out["disagreement_by_round"] = np.nanmean(np.nanmax(A, 2) - np.nanmin(A, 2), axis=0).round(2).tolist()   # mean max-min spread
    out["endpoint_share_by_round"] = np.nanmean((A == lo) | (A == hi), axis=(0, 2)).round(3).tolist()
    return out


def coding_quality(d, n_cpp=300):
    codes = collect(d)
    if not codes:
        return None
    by = defaultdict(lambda: {"py_ok": [], "len": [], "empty": 0, "n": 0})
    cpp = defaultdict(list)
    for run, rnd, code in codes:
        b = by[rnd]
        b["n"] += 1
        b["len"].append(len(code))
        b["empty"] += not code.strip()
        if run.startswith("py-"):
            try:
                ast.parse(code); b["py_ok"].append(1)
            except SyntaxError:
                b["py_ok"].append(0)
        elif len(cpp[rnd]) < n_cpp:
            with tempfile.NamedTemporaryFile("w", suffix=".cpp", delete=False) as f:
                f.write(code)
            ok = subprocess.run(["g++", "-fsyntax-only", "-fpermissive", "-w", f.name], capture_output=True, timeout=60).returncode == 0
            os.unlink(f.name)
            cpp[rnd].append(int(ok))
    return {str(r): {"codes": b["n"], "empty": b["empty"], "python_parses_%": 100 * float(np.mean(b["py_ok"])) if b["py_ok"] else None,
                     "cpp_syntax_ok_% (sample)": 100 * float(np.mean(cpp[r])) if cpp[r] else None,
                     "mean_chars": float(np.mean(b["len"]))} for r, b in sorted(by.items())}


def adversary_check(recs, runs):
    """hiring: score of the adversarial evaluator vs the other three, adv vs base condition, by round."""
    ag = AGENTS["hiring"]
    T = eff_T("hiring", recs, runs)
    if not T:
        return None
    S = score_tensor("hiring", recs, runs, T)
    acc = defaultdict(list)
    for r, a in S.items():
        m = runs[r]
        j = ag.index(m["adv_role"])
        acc[(m["cond"], "evaluator that is adversarial in 'adv'")].append(a[:, j])
        acc[(m["cond"], "other evaluators")].append(np.nanmean(np.delete(a, j, axis=1), axis=1))
    return {f"{c} | {k}": np.nanmean(np.stack(v), 0).round(3).tolist() for (c, k), v in sorted(acc.items())}


def md_table(rows, cols):
    f = lambda v: f"{v:.1f}" if isinstance(v, float) else str(v)
    return "\n".join(["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)] + ["| " + " | ".join(f(r[c]) for c in cols) + " |" for r in rows])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    a = ap.parse_args()
    out_dir = f"{a.run_dir}/analysis"
    os.makedirs(out_dir, exist_ok=True)
    report, md = {}, [f"# Quality report: {os.path.basename(os.path.abspath(a.run_dir))}\n"]
    for app in ("coding", "hiring", "review"):
        d = f"{a.run_dir}/{app}"
        if not os.path.isdir(d):
            continue
        recs, runs = load(d)
        if not recs:
            md.append(f"## {app}\nno replies saved yet\n")
            continue
        rd = rounds_done(app, recs, runs)
        h_rows, h_tot = health(recs)
        rep = {"health": h_rows, "totals": h_tot, "runs_total": len(runs),
               "runs_with_all_rounds_so_far": {str(t): sum(v >= t for v in rd.values()) for t in range(1, 6)}}
        md.append(f"## {app}\n\n{len(runs)} runs; runs with rounds 1..t complete: {rep['runs_with_all_rounds_so_far']}\n\n"
                  f"{h_tot['calls']} replies saved, {h_tot['gen_tokens']} generated tokens, "
                  f"{h_tot['gen_tok_per_s_overall']:.0f} gen tok/s overall (first to last reply)\n\n### Call health\n\n"
                  + md_table(h_rows, ["phase", "agent", "calls", "failed_%", "retried_%", "truncated_%", "prompt_too_long", "think_tok_mean", "gen_tok_mean", "gen_tok_p95"]) + "\n")
        sc = score_sanity(app, recs, runs)
        if sc:
            rep["scores"] = sc
            md.append("\n### Score sanity\n\n" + md_table([{"agent": k, "mean by round": v["mean_by_round"], "std by round": v["std_by_round"], "score counts": v["hist"]} for k, v in sc["per_agent"].items()],
                                                          ["agent", "mean by round", "std by round", "score counts"])
                      + f"\n\nMean max-min spread across agents by round: {sc['disagreement_by_round']}  \nShare of scores at the scale endpoints by round: {sc['endpoint_share_by_round']}\n")
        if app == "coding":
            cq = coding_quality(d)
            if cq:
                rep["code_validity"] = cq
                md.append("\n### Generated code validity\n\n" + md_table([{"round": r, **v} for r, v in cq.items()], ["round", "codes", "empty", "python_parses_%", "cpp_syntax_ok_% (sample)", "mean_chars"]) + "\n")
            cqr = f"{d}/codeql_results.jsonl"
            if os.path.exists(cqr):
                rows = [json.loads(l) for l in open(cqr)]
                rep["codeql"] = {"codes": len(rows), "with_alert_%": 100 * float(np.mean([bool(r["rules"]) for r in rows]))}
                md.append(f"\nCodeQL: {len(rows)} codes analysed, {rep['codeql']['with_alert_%']:.1f}% have at least one alert.\n")
        if app == "hiring":
            ac = adversary_check(recs, runs)
            if ac:
                rep["adversary_check"] = ac
                md.append("\n### Adversary check (mean score by round)\n\n" + "\n".join(f"- {k}: {v}" for k, v in ac.items()) + "\n")
        report[app] = rep
    sp = f"{out_dir}/summary.json"
    if os.path.exists(sp):
        s = json.load(open(sp))
        md.append("\n## Homogenization metrics (from summary.json)\n\n```json\n" + json.dumps({k: v for k, v in s.items() if k == "coverage" or (isinstance(v, dict) and "definition_rates" in v)}, indent=1) + "\n```\n")
    json.dump(report, open(f"{out_dir}/quality_report.json", "w"), indent=1, default=float)
    open(f"{out_dir}/quality_report.md", "w").write("\n".join(md))
    print("\n".join(md)[:6000])
