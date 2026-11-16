"""Local stand-in for the parts of vLLM that steering_agents.llm uses (LLM, get_tokenizer, enqueue, llm_engine.step,
SamplingParams), backed by a transformers model on MPS/CPU. vLLM itself does not install on macOS.

Used by tests/local_model_smoke.py (put this directory first on PYTHONPATH). Only for checking the code path with a real
tokenizer, chat template and model; not for results. Differences from vLLM: no prefix cache (num_cached_tokens = 0),
each step generates all queued requests as one padded batch with the first request's seed, probes run one at a time,
and logprobs are the raw log-softmax (as vLLM's default raw_logprobs). SHIM_MAX_NEW caps generated tokens (default 2500).
"""
import os, types
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

DEV = "mps" if torch.backends.mps.is_available() else "cpu"
MAX_NEW = int(os.environ.get("SHIM_MAX_NEW", 2500))


class SamplingParams:
    def __init__(self, max_tokens=16, temperature=1.0, top_p=1.0, top_k=-1, seed=None, logprobs=None, **kw):
        self.max_tokens, self.temperature, self.top_p, self.top_k = max_tokens, temperature, top_p, top_k
        self.seed, self.logprobs = seed, logprobs


class LLM:
    def __init__(self, model, **kw):
        self.tok = AutoTokenizer.from_pretrained(model)
        self.tok.padding_side = "left"
        self.model = AutoModelForCausalLM.from_pretrained(model, torch_dtype=torch.bfloat16).to(DEV).eval()
        self.queue, self.n, self.llm_engine = [], 0, self
        self.stats = {"gen": 0, "probe": 0}

    def get_tokenizer(self):
        return self.tok

    def enqueue(self, prompts, sps, use_tqdm=False):
        ids = []
        for p, sp in zip(prompts, sps):
            self.n += 1
            ids.append(f"{self.n}-shim{self.n:04d}")
            self.queue.append((str(self.n), p, sp))
        return ids

    def has_unfinished_requests(self):
        return bool(self.queue)

    def _out(self, rid, prompt_ids, gen_ids, finish, logprobs=None):
        text = self.tok.decode(gen_ids, skip_special_tokens=True)
        c = types.SimpleNamespace(text=text, token_ids=gen_ids, finish_reason=finish, logprobs=logprobs)
        return types.SimpleNamespace(request_id=rid, finished=True, outputs=[c], prompt_token_ids=prompt_ids,
                                     num_cached_tokens=0)

    @torch.no_grad()
    def step(self):
        cur, self.queue = self.queue, []
        outs = []
        probes = [x for x in cur if isinstance(x[1], dict)]
        gens = [x for x in cur if not isinstance(x[1], dict)]
        for rid, p, sp in probes:   # 1-token probe: top logprobs of the next token (raw, before temperature/top-k)
            ids = p["prompt_token_ids"]
            logits = self.model(torch.tensor([ids], device=DEV)).logits[0, -1].float()
            lp = torch.log_softmax(logits, -1)
            v, i = lp.topk(sp.logprobs or 1)
            top = {int(t): types.SimpleNamespace(logprob=float(x)) for t, x in zip(i.tolist(), v.tolist())}
            outs.append(self._out(rid, ids, [int(i[0])], "length", [top]))
            self.stats["probe"] += 1
        if gens:
            enc = self.tok([p for _, p, _ in gens], return_tensors="pt", padding=True, add_special_tokens=False).to(DEV)
            sp = gens[0][2]
            torch.manual_seed(sp.seed or 0)
            mx = min(MAX_NEW, max(s.max_tokens for _, _, s in gens))
            g = self.model.generate(**enc, max_new_tokens=mx, do_sample=True, temperature=sp.temperature,
                                    top_p=sp.top_p, top_k=sp.top_k, pad_token_id=self.tok.pad_token_id)
            L = enc["input_ids"].shape[1]
            for k, (rid, p, _) in enumerate(gens):
                prompt_ids = self.tok(p, add_special_tokens=False)["input_ids"]
                gen = g[k, L:].tolist()
                eos = {self.tok.eos_token_id, self.tok.pad_token_id, self.tok.convert_tokens_to_ids("<|im_end|>")}
                cut = next((j for j, t in enumerate(gen) if t in eos), None)
                finish = "stop" if cut is not None else "length"
                outs.append(self._out(rid, prompt_ids, gen[:cut] if cut is not None else gen, finish))
                self.stats["gen"] += 1
        return outs
