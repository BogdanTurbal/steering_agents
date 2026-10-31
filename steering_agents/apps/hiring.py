"""Hiring MAS: 4 evaluators (workers), no orchestrator (the broadcast state is the concatenated panel discussion)."""
import json
import math
import os
import random
import zlib

from ..jsonio import parse_fields
from ..mas import App, Call, Run, sys_user
from ..personas import persona
from .hiring_data import OCC, SIGNALS, GROUPS

DATA = os.path.join(os.path.dirname(__file__), "..", "..", "data", "hiring", "pairs.jsonl")
ROLES = {"subject": "Subject Matter Evaluator", "experience": "Experience Evaluator",
         "impact": "Impact Evaluator", "operational": "Operational Readiness Evaluator"}
ROLE_ORDER = list(ROLES)


def fill_signals(resume, group, variant):
    sig = SIGNALS[group][variant]
    for k, v in sig.items():
        resume = resume.replace(f"[{k}]", v)
    return resume


def stereotype_strings(occ):
    g, e = OCC[occ]
    return " > ".join(x.capitalize() for x in g), " > ".join(x.capitalize() for x in e)


class HiringApp(App):
    name = "hiring"
    scale = (1, 5)

    def __init__(self, T=5, adversary=False, adv_rounds=2, variants=5, adv_roles=None, data=DATA):
        """adversary: also run the 'adv' condition (one biased evaluator for rounds 1..adv_rounds, then its normal persona).
        variants: signal variants per demographic group (paper: 5 -> 40 resumes per pair)."""
        self.T, self.adversary, self.adv_rounds, self.variants, self.data = T, adversary, adv_rounds, variants, data
        self.adv_roles = adv_roles or ROLE_ORDER

    def runs(self, n_items, n_trials, seed=0):
        pairs = [json.loads(l) for l in open(self.data)]
        random.Random(seed).shuffle(pairs)
        runs = []
        for p in pairs[:n_items]:
            adv_role = self.adv_roles[zlib.crc32(p["pair_id"].encode()) % len(self.adv_roles)]
            for (eth, gen) in GROUPS:
                for v in range(self.variants):
                    item = {"id": f"{p['pair_id']}-{eth}-{gen}-v{v}", "pair": p["pair_id"], "occupation": p["occupation"],
                            "ethnicity": eth, "gender": gen, "variant": v, "adv_role": adv_role,
                            "jd": p["jd"], "resume": fill_signals(p["resume"], (eth, gen), v)}
                    for k in range(n_trials):
                        runs.append(Run(f"{item['id']}#t{k}#base", item, k, "base"))
                        if self.adversary:
                            runs.append(Run(f"{item['id']}#t{k}#adv", item, k, "adv"))
        return runs

    def phases(self, t):
        return ["review"]

    def calls(self, run, t, phase):
        it, hist = run.item, run.state.setdefault("hist", [])
        gs, es = stereotype_strings(it["occupation"])
        cs = []
        for role in ROLE_ORDER:
            adv = run.cond == "adv" and role == it["adv_role"] and t <= self.adv_rounds
            p = persona(ROLES[role] + (" (Adversarial)" if adv else ""))
            system = p["System Prompt"].replace("{gender_stereotype}", gs).replace("{ethnicity_stereotype}", es)
            body = f"## Job description\n{it['jd']}\n\n## Resume\n{it['resume']}"
            if t == 1:
                u = f"{p['Initial Prompt']}\n\n{body}\n\n{p['Initial Output Format Prompt']}"
            else:
                tr = "\n\n".join(f"### Round {k + 1}\n" + "\n".join(
                    f"**{ROLES[r]}** (score {int(o['score'])}): {o['reasoning']}" for r, o in rd.items()
                    if not o.get("failed")) for k, rd in enumerate(hist))
                u = (f"{p['Refinement Prompt']}\n\n{body}\n\n## Panel discussion so far\n{tr}\n\n"
                     f"{p['Refinement Output Format Prompt']}")
            u = f"Your name on this panel: {ROLES[role]}.\n\n{u}"
            cs.append(Call(run.id, t, phase, role, sys_user(system, u)))
        return cs

    def parse(self, call, text):
        r = parse_fields(text, str_fields=("reasoning",), num_fields=("score", "confidence"),
                         bounds={"score": (1, 5), "confidence": (1, 5)})
        if r is None:
            r = parse_fields(text, str_fields=("reasoning",), num_fields=("score",), bounds={"score": (1, 5)})
        return r

    def apply(self, run, t, phase, results):
        run.state.setdefault("hist", []).append(results)
