"""Peer-review MAS (structure of Sahu et al. 2025): 3 reviewers (workers); meta-reviewer + author (orchestrator)."""
import json
import os
import random

from ..jsonio import parse_fields
from ..mas import App, Call, Run, sys_user
from ..personas import persona

DATA = os.path.join(os.path.dirname(__file__), "..", "..", "data", "iclr2025", "papers_review.jsonl")
REVIEWERS = {"pragmatist": "Paper Reviewer Pragmatist", "theorist": "Paper Reviewer Theorist",
             "empiricist": "Paper Reviewer Empiricist"}
MAX_PAPER_CHARS = 400_000   # effectively no truncation (context is 262k tokens)


class ReviewApp(App):
    name = "review"
    scale = (1, 10)

    def __init__(self, T=5, data=DATA):
        self.T, self.data = T, data

    def runs(self, n_items, n_trials, seed=0):
        papers = [json.loads(l) for l in open(self.data)]
        random.Random(seed).shuffle(papers)
        return [Run(f"{p['paper_id']}#t{k}", {"id": p["paper_id"], "title": p["title"], "type": p["type"],
                                              "primary_area": p["primary_area"], "decision": p.get("decision", p["type"]), "text": p["text"][:MAX_PAPER_CHARS]}, k)
                for p in papers[:n_items] for k in range(n_trials)]

    def phases(self, t):
        return ["review", "orch"]

    @staticmethod
    def _discussion(run, who):
        """Discussion transcript. Reviewers see everything; author sees reviews + own rebuttals; meta sees reviews + own summaries."""
        s, out = run.state, []
        for k, rv in enumerate(s["rev"]):
            blk = [f"## Round {k + 1}"]
            for r, o in rv.items():
                if o.get("failed"):
                    continue
                blk.append(f"### Review by {r.capitalize()} reviewer (rating {o['score']:g})\n{o['output']}")
            if k < len(s["meta"]) and who in ("reviewer", "meta"):
                m = s["meta"][k]
                blk.append(f"### Meta-reviewer synthesis\n{m['output']}") if not m.get("failed") else None
            if k < len(s["auth"]) and who in ("reviewer", "author"):
                a = s["auth"][k]
                blk.append(f"### Author rebuttal\n{a['output']}") if not a.get("failed") else None
            out.append("\n\n".join(blk))
        return "\n\n".join(out)

    def calls(self, run, t, phase):
        s = run.state
        for k in ("rev", "meta", "auth"):
            s.setdefault(k, [])
        paper = f"# Paper: {run.item['title']}\n{run.item['text']}"
        if phase == "review":
            cs = []
            for role, pname in REVIEWERS.items():
                p = persona(pname)
                if t == 1:
                    u = f"{p['Initial Prompt']}\n\n{paper}\n\n{p['Initial Output Format Prompt']}\n\n{p['Trailing Output-Format Reminder']}"
                else:
                    u = (f"{p['Refinement Prompt']}\n\n{paper}\n\n# Discussion so far (you are the {role.capitalize()} reviewer)\n"
                         f"{self._discussion(run, 'reviewer')}\n\n{p['Refinement Output Format Prompt']}\n\n"
                         f"{p['Trailing Output-Format Reminder']}")
                cs.append(Call(run.id, t, phase, role, sys_user(p["System Prompt"], u)))
            return cs
        cs = []
        for role, pname, who in (("metareviewer", "Paper Metareviewer", "meta"), ("author", "Paper Author", "author")):
            p = persona(pname)
            u = (f"{p['Aggregation Prompt']}\n\n{paper}\n\n# Discussion so far\n{self._discussion(run, who)}\n\n"
                 f"{p['Trailing Output-Format Reminder']}")
            cs.append(Call(run.id, t, phase, role, sys_user(p["System Prompt"], u)))
        return cs

    def parse(self, call, text):
        if call.agent in REVIEWERS:
            return parse_fields(text, str_fields=("output",), num_fields=("score",), bounds={"score": (1, 10)})
        if call.agent == "metareviewer":
            r = parse_fields(text, str_fields=("output",), num_fields=("decision",), bounds={"decision": (1, 5)})
            return r or parse_fields(text, str_fields=("output",))
        return parse_fields(text, str_fields=("output",))

    def apply(self, run, t, phase, results):
        s = run.state
        for k in ("rev", "meta", "auth"):
            s.setdefault(k, [])
        if phase == "review":
            s["rev"].append(results)
        else:
            s["meta"].append(results["metareviewer"])
            s["auth"].append(results["author"])
