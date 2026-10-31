"""Code generation MAS: 4 reviewers (workers) + code generator (orchestrator), CodeLMSec stubs."""
import glob
import os
import random

from ..jsonio import parse_fields
from ..mas import App, Call, Run, sys_user
from ..personas import persona

ROOT = os.path.join(os.path.dirname(__file__), "..", "..", "data", "codelmsec", "benchmark", "benchmark_data")
REVIEWERS = {"security": "Code Reviewer Security", "optimality": "Code Reviewer Optimality",
             "correctness": "Code Reviewer Correctness", "readability": "Code Reviewer Readability"}
LANG = {"py": "Python", "c": "C++"}   # paper: Python and C++ (cpp-security-extended); stubs are C-style


def load_items():
    items = []
    for lang in ("py", "c"):
        for p in sorted(glob.glob(os.path.join(ROOT, lang, "CWE-*", "*"))):
            items.append({"id": f"{lang}-{os.path.basename(os.path.dirname(p))}-{os.path.basename(p).rsplit('.', 1)[0]}",
                          "lang": lang, "cwe": os.path.basename(os.path.dirname(p)), "stub": open(p).read()})
    return items


def fence(code, lang):
    return f"```{'python' if lang == 'py' else 'cpp'}\n{code}\n```"


class CodingApp(App):
    name = "coding"
    scale = (1, 5)

    def __init__(self, T=5, reviewers_see_all=True):
        """reviewers_see_all: reviewers get the broadcast global state (all codes + all four reviewers' feedback), as in
        Alg. 1. False = paper-literal reading of "the critic's feedback": a reviewer sees only its own past reviews."""
        self.T, self.reviewers_see_all = T, reviewers_see_all

    def runs(self, n_items, n_trials, seed=0):
        items = load_items()
        random.Random(seed).shuffle(items)
        return [Run(f"{it['id']}#t{k}", it, k) for it in items[:n_items] for k in range(n_trials)]

    def phases(self, t):
        return (["init"] if t == 1 else []) + ["review", "orch"]

    def _hist(self, run, own=None):
        """History shown to a party: stub, previous codes, feedback (own reviewer only, or all critics)."""
        s = run.state
        out = [f"# Original code stub\n{fence(run.item['stub'], run.item['lang'])}"]
        for k in range(len(s["rev"])):   # completed review rounds
            out.append(f"# Round {k + 1} code\n{fence(s['codes'][k], run.item['lang'])}")
            fb = [(own, s["rev"][k][own])] if own else list(s["rev"][k].items())
            for who, r in fb:
                out.append(f"## Round {k + 1} feedback ({who} reviewer, deployability score {r.get('score', 'n/a')})\n"
                           f"{r.get('output', '(no review)')}")
        return "\n\n".join(out)

    def calls(self, run, t, phase):
        s = run.state
        s.setdefault("codes", [])
        s.setdefault("rev", [])
        lang = LANG[run.item["lang"]]
        if phase == "init":
            p = persona("Code Generator")
            u = f"{p['Initial Prompt']}\n\n{fence(run.item['stub'], run.item['lang'])}\n\n{p['Initial Output Format Prompt']}"
            return [Call(run.id, t, phase, "generator", sys_user(p["System Prompt"].replace("{language}", lang), u))]
        if phase == "review":
            code = s["codes"][t - 1]
            cs = []
            for role, pname in REVIEWERS.items():
                p = persona(pname)
                if t == 1:
                    u = (f"{p['Initial Prompt']}\n\n# Original code stub\n{fence(run.item['stub'], run.item['lang'])}\n\n"
                         f"# Complete code to review\n{fence(code, run.item['lang'])}\n\n{p['Initial Output Format Prompt']}")
                else:
                    hist = self._hist(run, own=None if self.reviewers_see_all else role)
                    u = (f"{p['Refinement Prompt']}\n\n(You are the {role} reviewer.)\n\n{hist}\n\n"
                         f"# Latest round (round {t}) complete code to review\n{fence(code, run.item['lang'])}\n\n"
                         f"{p['Refinement Output Format Prompt']}")
                cs.append(Call(run.id, t, phase, role, sys_user(p["System Prompt"].replace("{language}", lang), u)))
            return cs
        p = persona("Code Generator")   # orch: rewrite given all critics' feedback of round t
        hist = self._hist(run) if len(s["rev"]) >= t else None
        u = f"{p['Refinement Prompt']}\n\n{hist}\n\n{p['Refinement Output Format Prompt']}"
        return [Call(run.id, t, phase, "generator", sys_user(p["System Prompt"].replace("{language}", lang), u))]

    def parse(self, call, text):
        if call.agent == "generator":
            r = parse_fields(text, str_fields=("output",))
            if r is None:   # fallback: a fenced code block
                import re
                m = re.search(r"```(?:python|cpp|c\+\+|c)?\n(.*?)```", text, re.S)
                return {"output": m.group(1)} if m else None
            return r
        return parse_fields(text, str_fields=("output",), num_fields=("score",), bounds={"score": (1, 5)})

    def apply(self, run, t, phase, results):
        s = run.state
        s.setdefault("codes", [])
        s.setdefault("rev", [])
        if phase in ("init", "orch"):
            g = results["generator"]
            if g.get("failed"):   # keep previous code if the generator failed
                g = {"output": s["codes"][-1] if s["codes"] else run.item["stub"]}
            s["codes"].append(g["output"])
        else:
            s["rev"].append(results)
