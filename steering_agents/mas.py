"""Orchestrator-driven MAS (paper Alg. 1) executed as batched, resumable phases.

An App defines runs (item x trial [x condition]), per-round phases (e.g. worker `review`, then orchestrator `orch`),
the chat calls of each phase, how to parse an agent's reply, and how the parsed replies update the run state.
The Runner submits every call of a phase across a wave of runs to vLLM at once, appends each reply to records.jsonl as
soon as it finishes, re-samples unparsable replies with new seeds and resumes by replaying stored parsed replies through
App.apply.
"""
import json
import os
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from .llm import seed_for


@dataclass
class Call:
    run: str
    t: int
    phase: str
    agent: str
    messages: list
    @property
    def key(self):
        return f"{self.run}|{self.t}|{self.phase}|{self.agent}"


@dataclass
class Run:
    id: str
    item: dict
    trial: int
    cond: str = "base"
    state: dict = field(default_factory=dict)
    @property
    def meta(self):
        m = {k: v for k, v in self.item.items() if k not in ("text", "stub", "jd", "resume")}
        return {"run": self.id, "trial": self.trial, "cond": self.cond, **m}


class App(ABC):
    name: str
    T: int = 5
    scale: tuple = (1, 5)

    @abstractmethod
    def runs(self, n_items, n_trials, seed): ...
    @abstractmethod
    def phases(self, t): ...
    @abstractmethod
    def calls(self, run, t, phase): ...
    @abstractmethod
    def parse(self, call, text): ...        # dict or None
    @abstractmethod
    def apply(self, run, t, phase, results): ...   # results: agent -> parsed dict (or {"failed": True})


def sys_user(system, user):
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


class Runner:
    def __init__(self, app, engine, out_dir, shard=(0, 1), wave=192, retries=2, log=print):
        self.app, self.eng, self.out, self.shard = app, engine, out_dir, shard
        self.wave, self.retries, self.log = wave, retries, log
        os.makedirs(out_dir, exist_ok=True)
        self.rec_path = os.path.join(out_dir, f"records_shard{shard[0]}of{shard[1]}.jsonl")
        self.done = {}
        if os.path.exists(self.rec_path):
            for ln in open(self.rec_path):
                try:
                    r = json.loads(ln)
                except json.JSONDecodeError:   # torn last line after a kill
                    continue
                self.done[r["key"]] = r

    def _generate(self, calls):
        """Fill self.done for calls. Every reply is appended (and flushed) to records.jsonl the moment it finishes, so an
        interrupted job loses only the calls that were still generating. Unparsable replies are re-sampled in a new pass."""
        pending = {c.key: c for c in calls if c.key not in self.done}
        attempts = {k: 0 for k in pending}
        while pending:
            keys = list(pending)
            nxt, n, ntok, t0 = {}, 0, 0, time.time()
            batch = [(pending[k].messages, seed_for(k, attempts[k])) for k in keys]
            with open(self.rec_path, "a", buffering=1) as f:
                for i, o in self.eng.stream(batch):
                    k, c = keys[i], pending[keys[i]]
                    parsed = self.app.parse(c, o.answer) if o.answer else None
                    attempts[k] += 1
                    n, ntok = n + 1, ntok + o.n_tokens
                    if parsed is None and attempts[k] <= self.retries and o.finish != "prompt_too_long":
                        nxt[k] = c
                    else:
                        rec = {"key": k, "run": c.run, "t": c.t, "phase": c.phase, "agent": c.agent,
                               "parsed": parsed if parsed is not None else {"failed": True},
                               "answer": o.answer, "think_tokens": o.think_tokens, "n_tokens": o.n_tokens,
                               "finish": o.finish, "attempts": attempts[k], "time": round(time.time(), 1)}
                        self.done[k] = rec
                        f.write(json.dumps(rec) + "\n")
                    if n % 200 == 0:
                        dt = time.time() - t0
                        self.log(f"  {n}/{len(keys)} calls, {ntok / dt:.0f} gen tok/s, retry-queue {len(nxt)}")
            self.log(f"  pass done: {n} calls, {ntok} gen tok, {time.time() - t0:.0f}s, retry-queue {len(nxt)}")
            pending = nxt

    def run(self, runs):
        runs = sorted(runs, key=lambda r: r.id)[self.shard[0]::self.shard[1]]
        with open(os.path.join(self.out, f"runs_shard{self.shard[0]}of{self.shard[1]}.jsonl"), "w") as f:
            for r in runs:
                f.write(json.dumps(r.meta) + "\n")
        for w in range(0, len(runs), self.wave):
            wave = runs[w:w + self.wave]
            self.log(f"wave {w // self.wave + 1}/{-(-len(runs) // self.wave)}: {len(wave)} runs")
            for t in range(1, self.app.T + 1):
                for phase in self.app.phases(t):
                    per_run = {r.id: self.app.calls(r, t, phase) for r in wave}
                    self.log(f" round {t} phase {phase}: {sum(map(len, per_run.values()))} calls")
                    self._generate([c for cs in per_run.values() for c in cs])
                    for r in wave:
                        self.app.apply(r, t, phase, {c.agent: self.done[c.key]["parsed"] for c in per_run[r.id]})
