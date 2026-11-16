"""vLLM engine for the agent model (Qwen3.8-27B, bitsandbytes 4-bit) with thinking."""
import glob
import os
import zlib
from dataclasses import dataclass

from . import confidence as CF

HF_HOME = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
# Default: the full bf16 checkpoint, no quantization. quant="bitsandbytes" gives in-flight 4-bit loading, which decodes slowly
# (~12-16 tok/s per sequence in a small test). The pre-quantized unsloth checkpoint (unsloth/Qwen3.8-27B-unsloth-bnb-4bit)
# does not load in vLLM 0.25.1: bitsandbytes_loader fails on `assert param_data.shape == loaded_weight.shape`.
DEFAULT_MODEL = "Qwen/Qwen3.8-27B"


def resolve_model(name):
    if os.path.isdir(name):
        return name
    snaps = sorted(glob.glob(os.path.join(HF_HOME, "models--" + name.replace("/", "--"), "snapshots", "*")))
    if not snaps:
        raise FileNotFoundError(f"{name} not in {HF_HOME}; download it on the login node")
    return snaps[-1]


def seed_for(*parts):
    return zlib.crc32("|".join(map(str, parts)).encode()) & 0x7FFFFFFF


@dataclass
class Out:
    answer: str | None   # text after </think>; None if thinking never closed (truncated)
    think_tokens: int
    n_tokens: int
    finish: str
    conf: dict | None = None   # token-likelihood confidence (steering_agents/confidence.py), only if probes were requested


def split_think(text):
    if "</think>" in text:
        return text.rsplit("</think>", 1)[1].strip(), text.split("</think>")[0]
    return None, text


def prepare_vllm_env():
    """Cluster-safe vLLM defaults, set before the engine is created (existing values are respected)."""
    os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")   # the FlashInfer JIT sampler fails on nodes without curand.h
    os.environ.setdefault("VLLM_BATCH_INVARIANT", "0")
    os.environ.setdefault("VLLM_HOST_IP", "127.0.0.1")


class Engine:
    """vLLM wrapper for Qwen3.8 with thinking. LLM() settings: dtype auto, tensor parallel 1, GPU memory 0.90,
    trust_remote_code, CUDA graphs on, gdn_prefill_backend triton, max_model_len 262144, max_num_seqs 1000, prefix caching
    (long shared prompt prefixes). Prompts are rendered with tokenizer.apply_chat_template(enable_thinking=True,
    reasoning_effort=...) and fed to vLLM as text; sampling is temperature 1.0, top_p 0.95, top_k 20 with a per-request seed."""

    def __init__(self, model=DEFAULT_MODEL, max_model_len=262144, gpu_mem=0.90, effort="medium", max_tokens=12288,
                 max_num_seqs=1000, quant=None, gdn_prefill_backend="triton"):
        prepare_vllm_env()
        from vllm import LLM
        self.effort, self.max_tokens = effort, max_tokens
        kw = dict(model=resolve_model(model), dtype="auto", tensor_parallel_size=1, gpu_memory_utilization=gpu_mem,
                  trust_remote_code=True, max_model_len=max_model_len, enforce_eager=False, download_dir=None,
                  enable_prefix_caching=True, max_num_seqs=max_num_seqs)
        if gdn_prefill_backend:
            kw["gdn_prefill_backend"] = gdn_prefill_backend
        if quant:
            kw["quantization"] = quant
            kw["load_format"] = quant
        self.max_model_len = max_model_len
        self.llm = LLM(**kw)
        self.tok = self.llm.get_tokenizer()

    def render(self, messages, thinking=True):
        """Chat template -> text (drops reasoning_effort if the template rejects it)."""
        kw = dict(tokenize=False, add_generation_prompt=True, enable_thinking=thinking, reasoning_effort=self.effort)
        try:
            return self.tok.apply_chat_template(messages, **kw)
        except TypeError:
            kw.pop("reasoning_effort")
            return self.tok.apply_chat_template(messages, **kw)

    def stream(self, batch, probes=None):
        """batch: list of (messages, seed). Yields (index, Out) as soon as each request finishes (so callers can persist
        every reply immediately). Uses vLLM's enqueue + engine.step loop (what LLM.generate does internally).
        probes: optional list (one per batch item) of {field: K} or None. When a reply finishes, 1-token probes at those
        fields are enqueued right away (their prefix is still in the prefix cache) and the reply is yielded with Out.conf
        once all its probes are back (see confidence.py)."""
        from vllm import SamplingParams
        texts = [self.render(m) for m, _ in batch]
        plen = [len(self.tok.encode(t, add_special_tokens=False)) for t in texts]
        # Prompts that cannot fit the context (minus room for >= 2k generated tokens) fail alone: vLLM would otherwise
        # raise for the whole batch.
        ok = []
        for i, n in enumerate(plen):
            if n + 2048 <= self.max_model_len:
                ok.append(i)
            else:
                yield i, Out(None, 0, 0, "prompt_too_long")
        if not ok:
            return
        sps = [SamplingParams(max_tokens=min(self.max_tokens, self.max_model_len - plen[i] - 64), seed=batch[i][1],
                              temperature=1.0, top_p=0.95, top_k=20) for i in ok]
        ids = self.llm.enqueue([texts[i] for i in ok], sps, use_tqdm=False)
        # enqueue returns INTERNAL ids ("<counter>-<8 random chars>", vLLM input_processor.assign_request_id), while
        # RequestOutput.request_id is the external "<counter>": key by the external part.
        id2i = {rid.rsplit("-", 1)[0]: i for rid, i in zip(ids, ok)}
        probe_of = {}   # probe request id -> (batch index, probe name)
        waiting = {}    # batch index -> [Out, spec, {probe name: top logprobs}, n probes left, cached tokens, prompt tokens,
                        #                {field: number written}]
        eng = self.llm.llm_engine
        while eng.has_unfinished_requests():
            for o in eng.step():
                if not o.finished:
                    continue
                rid = o.request_id if (o.request_id in id2i or o.request_id in probe_of) else o.request_id.rsplit("-", 1)[0]
                if rid in id2i:
                    i, out = id2i[rid], self._out(o)
                    spec = probes[i] if probes else None
                    pre, written = self._probe_prefixes(batch[i][0], o, spec) if spec and out.answer else ({}, {})
                    if not pre:
                        yield i, out
                        continue
                    names = list(pre)
                    psp = SamplingParams(max_tokens=1, temperature=1.0, logprobs=CF.TOP_LOGPROBS)
                    pids = self.llm.enqueue([{"prompt_token_ids": pre[n]} for n in names], [psp] * len(names), use_tqdm=False)
                    for pid, n in zip(pids, names):
                        probe_of[pid.rsplit("-", 1)[0]] = (i, n)
                    waiting[i] = [out, spec, {}, len(names), 0, 0, written]
                else:
                    i, name = probe_of.pop(rid)
                    w = waiting[i]
                    lp = o.outputs[0].logprobs
                    if lp:
                        w[2][name] = {t: x.logprob for t, x in lp[0].items()}
                    w[3] -= 1
                    w[4] += getattr(o, "num_cached_tokens", 0) or 0
                    w[5] += len(o.prompt_token_ids or [])
                    if w[3] == 0:
                        del waiting[i]
                        w[0].conf = CF.assemble(w[1], w[2], self.digits, round(w[4] / w[5], 3) if w[5] else None, w[6])
                        yield i, w[0]

    def _probe_prefixes(self, messages, o, spec):
        """Token prefixes of the probes for one finished reply (confidence.probe_prefixes); C1n uses the same messages
        rendered with thinking disabled plus a forced `{"score": `."""
        if not hasattr(self, "digits"):
            self.digits, self.ten_ok = CF.digit_ids(self.tok), CF.splits_ten(self.tok)
            if not self.ten_ok:
                print('WARNING: the tokenizer writes "10" as one token; 1-10 scores are not probed', flush=True)
        nothink = None
        if "score" in spec:
            nothink = self.tok.encode(self.render(messages, thinking=False) + CF.NOTHINK_PREFIX, add_special_tokens=False)
        pre, written = CF.probe_prefixes(self.tok, list(o.prompt_token_ids), list(o.outputs[0].token_ids), spec, nothink,
                                         self.ten_ok)
        return {n: x for n, x in pre.items() if len(x) < self.max_model_len - 1}, written   # too long fails the whole enqueue

    def chat(self, batch):
        """Blocking variant: list[Out] in input order."""
        res = [None] * len(batch)
        for i, o in self.stream(batch):
            res[i] = o
        return res

    def _out(self, o):
        c = o.outputs[0]
        ans, think = split_think(c.text)
        return Out(ans, len(self.tok.encode(think)) if think else 0, len(c.token_ids), c.finish_reason)
