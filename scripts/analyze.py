"""Metrics, tables and figures for one experiment dir (<run>/{coding,hiring,review}); writes <run>/analysis/.

  python scripts/analyze.py --run-dir experiments/<run>
Mirrors the paper: Fig 3 (conformity/polarization/inertia), Fig 4 (code vulnerabilities, deployed-only), Fig 5
(hiring bias with adversary), Fig 6 (peer-review area means), Fig 7 (cross-trial variance, code similarity).
"""
import argparse, glob, json, os, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from steering_agents import metrics as M
from steering_agents.apps.hiring_data import OCC

AGENTS = {"coding": ["security", "optimality", "correctness", "readability"],
          "hiring": ["subject", "experience", "impact", "operational"],
          "review": ["pragmatist", "theorist", "empiricist"]}
SCALE = {"coding": (1, 5), "hiring": (1, 5), "review": (1, 10)}
C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]   # fixed categorical order
INK, GRID = "#33332f", "#e3e2dc"


def style(ax, title=None, xl=None, yl=None):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(axis="y", color=GRID, lw=0.6)
    ax.tick_params(colors=INK, labelsize=8)
    if title: ax.set_title(title, fontsize=10, color=INK, loc="left")
    if xl: ax.set_xlabel(xl, fontsize=8, color=INK)
    if yl: ax.set_ylabel(yl, fontsize=8, color=INK)


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


def rounds_done(app, recs, runs):
    """{run: number of consecutive rounds 1..t for which every worker has a saved reply}."""
    ag = AGENTS[app]
    out = {}
    for run in runs:
        t = 0
        while all(f"{run}|{t + 1}|review|{a}" in recs for a in ag):
            t += 1
        out[run] = t
    return out


def eff_T(app, recs, runs, max_T=5, min_runs=4):
    """Deepest round T such that enough runs completed rounds 1..T (so interrupted jobs still give metrics on what finished)."""
    rd = rounds_done(app, recs, runs)
    for T in range(max_T, 0, -1):
        if sum(v >= T for v in rd.values()) >= max(min_runs, 0.02 * len(rd)):
            return T
    return 0


def score_tensor(app, recs, runs, T):
    """{run: array [T, A]} (NaN where missing) for runs with all T rounds present."""
    ag = AGENTS[app]
    S = {}
    for run in runs:
        a = np.full((T, len(ag)), np.nan)
        for t in range(1, T + 1):
            for j, ag_ in enumerate(ag):
                r = recs.get(f"{run}|{t}|review|{ag_}")
                if r and not r["parsed"].get("failed"):
                    a[t - 1, j] = r["parsed"]["score"]
        if not np.isnan(a).all(axis=1).any():
            S[run] = a
    return S


def per_trial(fn, S, runs, n_trials):
    """Apply fn(array [R,T,A]) per trial index; returns [M, T or T-1]."""
    out = []
    for k in range(n_trials):
        ids = [r for r in S if runs[r]["trial"] == k]
        if ids: out.append(fn(np.stack([S[r] for r in ids])))
    return np.array(out)


def band(ax, y, color, label, x0=1):
    m = y.mean(0)
    ci = 1.96 * y.std(0, ddof=1) / np.sqrt(len(y)) if len(y) > 1 else np.zeros_like(m)
    x = np.arange(x0, x0 + len(m))
    ax.plot(x, m, color=color, lw=2, label=label)
    ax.fill_between(x, m - ci, m + ci, color=color, alpha=0.15, lw=0)
    ax.set_xticks(x)


def dynamics(app, d, out, summary, cond=None):
    recs, runs = load(f"{d}/{app}")
    if cond: runs = {k: v for k, v in runs.items() if v["cond"] == cond}
    T = eff_T(app, recs, runs)
    if not T: return None   # nothing complete yet
    S = score_tensor(app, recs, runs, T)
    if not S: return None
    summary.setdefault("coverage", {})[f"{app}{'_' + cond if cond else ''}"] = {"runs_total": len(runs), "runs_analysed": len(S), "rounds_analysed": T}
    n_trials = 1 + max(runs[r]["trial"] for r in S)
    lo, hi = SCALE[app]
    stack = np.stack(list(S.values()))
    summary[f"{app}{'_' + cond if cond else ''}"] = {"runs": len(S), "definition_rates": M.definition_rates(stack, lo, hi),
        "conformity": M.conformity_curve(stack).tolist(), "polarization": M.polarization_curve(stack, lo, hi).tolist(),
        "inertia": M.inertia_curve(stack).tolist()}
    fig, ax = plt.subplots(1, 3, figsize=(11, 3))
    band(ax[0], per_trial(M.conformity_curve, S, runs, n_trials), C[0], app)
    band(ax[1], per_trial(lambda s: M.polarization_curve(s, lo, hi), S, runs, n_trials), C[0], app)
    band(ax[2], per_trial(M.inertia_curve, S, runs, n_trials), C[0], app, x0=1.5)
    style(ax[0], "Conformity", "round", "inter-agent variance")
    style(ax[1], "Polarization", "round", "distance from endpoints")
    style(ax[2], "Inertia", "round transition (t to t+1)", "change in panel mean")
    fig.tight_layout(); fig.savefig(f"{out}/fig3_{app}{'_' + cond if cond else ''}.png", dpi=150); plt.close(fig)
    return S, runs, recs


def variance_over_trials(S, runs, T):
    byitem = {}
    for r, a in S.items():
        byitem.setdefault(runs[r]["id"], []).append(np.nanmean(a, axis=1))
    v = [np.var(np.stack(x), axis=0, ddof=1) for x in byitem.values() if len(x) > 1]
    return np.mean(v, axis=0) if v else None


def coding(d, out, summary):
    res = dynamics("coding", d, out, summary)
    if not res: return
    S, runs, recs = res
    T = next(iter(S.values())).shape[0]
    p = f"{d}/coding/codeql_results.jsonl"
    if os.path.exists(p):
        rows = [json.loads(l) for l in open(p)]
        dep = {(r, t): np.nanmean(S[r][t - 1]) >= 3 for r in S for t in range(1, T + 1)}
        rows = [r for r in rows if r["run"] in S and r["round"] <= T]
        rules = {}
        for r in rows:
            for u in r["rules"]: rules[u] = rules.get(u, 0) + 1
        top = [u for u, _ in sorted(rules.items(), key=lambda x: -x[1])[:10]]
        fig, ax = plt.subplots(1, 2, figsize=(12, 3.4))
        for k, (title, filt) in enumerate((("All generated code", lambda r: True), ("Deployed code only (panel mean >= 3)", lambda r: dep[(r["run"], r["round"])]))):
            for t in range(1, T + 1):
                rs = [r for r in rows if r["round"] == t]
                y = [100 * sum(u in r["rules"] and filt(r) for r in rs) / max(1, len(rs)) for u in top]
                ax[k].bar(np.arange(len(top)) + (t - 1) * 0.8 / T, y, width=0.8 / T, color=C[t - 1], label=f"round {t}")
            ax[k].set_xticks(np.arange(len(top)) + 0.4); ax[k].set_xticklabels(top, rotation=40, ha="right", fontsize=7)
            style(ax[k], title, None, "% of all samples")
        ax[0].legend(fontsize=7, frameon=False)
        fig.tight_layout(); fig.savefig(f"{out}/fig4_vulnerabilities.png", dpi=150); plt.close(fig)
        summary["coding_vuln_rate_by_round"] = [float(np.mean([bool(r["rules"]) for r in rows if r["round"] == t])) for t in range(1, T + 1)]
        summary["coding_top_rules"] = top
        # --- reviewer confidence vs actual security -------------------------------------------------------------------
        # pass = code with NO CodeQL alert. Distance to endpoints (paper's polarization) = reviewers' confidence collapsing;
        # the expectation is that it falls much faster than the real alert rate does (blind spot).
        stack = np.stack(list(S.values()))
        dist = M.polarization_curve(stack, 1, 5)
        panel = np.nanmean(stack, axis=(0, 2))
        def alert_rate(lang=None):
            return np.array([np.mean([bool(r["rules"]) for r in rows if r["round"] == t and (lang is None or r["lang"] == lang)] or [np.nan])
                             for t in range(1, T + 1)])
        ar, ar_py, ar_c = alert_rate(), alert_rate("py"), alert_rate("c")
        summary["coding_security_vs_confidence"] = {
            "pass_rate_all": (1 - ar).tolist(), "pass_rate_python": (1 - ar_py).tolist(), "pass_rate_c": (1 - ar_c).tolist(),
            "alert_rate_all": ar.tolist(), "alert_rate_python": ar_py.tolist(),
            "deployable_share": [float(np.mean([dep[(r, t)] for r in S])) for t in range(1, T + 1)],
            "panel_mean_score": panel.tolist(), "perceived_readiness_(mean-1)/4": ((panel - 1) / 4).tolist(),
            "distance_to_endpoints": dist.tolist(),
            "distance_to_endpoints_relative_to_round1": (dist / dist[0]).tolist(),
            "alert_rate_relative_to_round1": (ar / ar[0]).tolist(),
            "alert_rate_python_relative_to_round1": (ar_py / ar_py[0]).tolist()}
        x = np.arange(1, T + 1)
        fig, ax = plt.subplots(1, 2, figsize=(10.5, 3.6))
        ax[0].plot(x, dist / dist[0], color=C[0], lw=2, marker="o", ms=4, label="reviewers: distance to endpoints")
        ax[0].plot(x, ar / ar[0], color=C[1], lw=2, marker="o", ms=4, label="actual: share of code with a CodeQL alert")
        ax[0].plot(x, ar_py / ar_py[0], color=C[1], lw=1.5, ls="--", label="actual, Python only")
        ax[0].set_ylim(0, 1.05); ax[0].set_xticks(x); ax[0].legend(fontsize=7, frameon=False)
        style(ax[0], "Both indexed to round 1 (=1)", "round", "value / round-1 value")
        ax[1].plot(x, (panel - 1) / 4, color=C[0], lw=2, marker="o", ms=4, label="reviewers: panel mean score, scaled (score-1)/4")
        ax[1].plot(x, 1 - ar, color=C[1], lw=2, marker="o", ms=4, label="actual: pass rate (no CodeQL alert)")
        ax[1].plot(x, 1 - ar_py, color=C[1], lw=1.5, ls="--", label="actual pass rate, Python only")
        ax[1].set_ylim(0, 1.05); ax[1].set_xticks(x); ax[1].legend(fontsize=7, frameon=False, loc="lower right")
        style(ax[1], "Perceived readiness vs actual pass rate", "round", "fraction")
        fig.tight_layout(); fig.savefig(f"{out}/fig7_code_confidence_vs_security.png", dpi=150); plt.close(fig)
    sp = f"{d}/coding/code_similarity.jsonl"
    if os.path.exists(sp):
        rows = [json.loads(l) for l in open(sp)]
        summary["code_difference_by_round"] = [float(1 - np.mean([r["same"] for r in rows if r["round"] == t] or [np.nan])) for t in range(1, T + 1)]


def hiring(d, out, summary):
    for cond in ("base", "adv"):
        dynamics("hiring", d, out, summary, cond)
    recs, runs = load(f"{d}/hiring")
    T = eff_T("hiring", recs, runs)
    if not T: return
    S = score_tensor("hiring", recs, runs, T)
    if not S: return
    ag = AGENTS["hiring"]
    # bias = score(most privileged group) - score(least privileged), per occupation order, per (pair, trial, cond)
    def bias(selector):
        res = {}
        for cond in ("base", "adv"):
            groups = {}
            for r, a in S.items():
                m = runs[r]
                if m["cond"] != cond: continue
                adv = ag.index(m["adv_role"])
                val = selector(a, adv)
                if val is None: continue
                g_ord, e_ord = OCC[m["occupation"]]
                groups.setdefault((m["pair"], m["trial"]), {"e": {}, "g": {}})
                groups[(m["pair"], m["trial"])]["e"].setdefault(m["ethnicity"], []).append(val)
                groups[(m["pair"], m["trial"])]["g"].setdefault(m["gender"], []).append(val)
            for kind, idx in (("ethnicity", "e"), ("gender", "g")):
                bs = []
                for (pair, trial), gr in groups.items():
                    occ = [m["occupation"] for m in runs.values() if m["pair"] == pair][0]
                    order = OCC[occ][1] if kind == "ethnicity" else OCC[occ][0]
                    if order[0] in gr[idx] and order[-1] in gr[idx]:
                        bs.append(np.mean(gr[idx][order[0]], axis=0) - np.mean(gr[idx][order[-1]], axis=0))
                if bs: res[(cond, kind)] = np.mean(bs, axis=0)
        return res
    sel = {"panel mean": lambda a, adv: np.nanmean(a, axis=1),
           "adversary": lambda a, adv: a[:, adv],
           "other agents": lambda a, adv: np.nanmean(np.delete(a, adv, axis=1), axis=1)}
    summary["hiring_bias"] = {}
    for kind in ("ethnicity", "gender"):
        fig, ax = plt.subplots(figsize=(5, 3.2))
        for i, (name, f) in enumerate(sel.items()):
            b = bias(f)
            if ("adv", kind) in b:
                ax.plot(range(1, T + 1), b[("adv", kind)], color=C[i], lw=2, label=f"{name} (with adversary)")
                summary["hiring_bias"][f"{kind}:{name}:adv"] = b[("adv", kind)].tolist()
        b = bias(sel["panel mean"])
        if ("base", kind) in b:
            ax.plot(range(1, T + 1), b[("base", kind)], color=INK, lw=2, ls="--", label="all agents (no adversary)")
            summary["hiring_bias"][f"{kind}:panel mean:base"] = b[("base", kind)].tolist()
        style(ax, f"{kind.capitalize()} bias (most - least privileged)", "round", "score difference")
        ax.legend(fontsize=7, frameon=False)
        fig.tight_layout(); fig.savefig(f"{out}/fig5_bias_{kind}.png", dpi=150); plt.close(fig)


def review(d, out, summary):
    res = dynamics("review", d, out, summary)
    if not res: return
    S, runs, recs = res
    T = next(iter(S.values())).shape[0]
    areas = {}
    for r, a in S.items():
        areas.setdefault(runs[r]["primary_area"], []).append(np.nanmean(a, axis=1))
    areas = {k: np.mean(v, axis=0) for k, v in areas.items() if len(v) >= 5}
    if not areas: return
    cm = np.mean(list(areas.values()), axis=0)
    fig, ax = plt.subplots(figsize=(7, 4))
    spread = np.array([a for a in areas.values()])
    for k, a in areas.items():
        ax.plot(range(1, T + 1), a - cm, color=GRID, lw=1)
    ends = sorted(areas, key=lambda k: areas[k][-1] - cm[-1])
    for i, k in enumerate(ends[:2] + ends[-2:]):
        ax.plot(range(1, T + 1), areas[k] - cm, color=C[i], lw=2, label=k[:45])
    style(ax, "Mean reviewer score by primary area (centered)", "round", "score - cross-area mean")
    ax.legend(fontsize=6, frameon=False)
    fig.tight_layout(); fig.savefig(f"{out}/fig6_area_means.png", dpi=150); plt.close(fig)
    dec = {}
    for r, a in S.items():
        dec.setdefault(runs[r]["decision"], []).append(np.nanmean(a, axis=1))
    summary["review_mean_score_by_decision_by_round"] = {k: np.mean(v, axis=0).tolist() for k, v in dec.items()}
    summary["review_area_spread_by_round"] = (spread.max(0) - spread.min(0)).tolist()
    summary["review_areas_n"] = len(areas)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--partial", action="store_true", help="accepted for clarity: partial results are always handled (see eff_T)")
    a = ap.parse_args()
    out = f"{a.run_dir}/analysis"
    os.makedirs(out, exist_ok=True)
    summary = {}
    for app, fn in (("coding", coding), ("hiring", hiring), ("review", review)):
        if os.path.isdir(f"{a.run_dir}/{app}"):
            fn(a.run_dir, out, summary)
            # Fig 7: cross-trial variance of the panel mean (surface-level variance from sampling)
            recs, runs = load(f"{a.run_dir}/{app}")
            if recs:
                base = {k: v for k, v in runs.items() if v["cond"] == "base"}
                T = eff_T(app, recs, base)
                S = score_tensor(app, recs, base, T) if T else {}
                v = variance_over_trials(S, runs, T) if S else None
                if v is not None: summary[f"{app}_var_over_trials"] = v.tolist()
    json.dump(summary, open(f"{out}/summary.json", "w"), indent=1)
    print(json.dumps({k: (v if not isinstance(v, dict) else {kk: vv for kk, vv in v.items() if kk == "definition_rates" or kk == "runs"}) for k, v in summary.items()}, indent=1)[:3000])
