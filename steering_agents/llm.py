"""vLLM engine for the agent model (Qwen3.8-27B, bitsandbytes 4-bit) with thinking."""
import glob
import os
import zlib
from dataclasses import dataclass

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

    def render(self, messages):
        """Chat template -> text (drops reasoning_effort if the template rejects it)."""
        kw = dict(tokenize=False, add_generation_prompt=True, enable_thinking=True, reasoning_effort=self.effort)
        try:
            return self.tok.apply_chat_template(messages, **kw)
        except TypeError:
            kw.pop("reasoning_effort")
            return self.tok.apply_chat_template(messages, **kw)

    def stream(self, batch):
        """batch: list of (messages, seed). Yields (index, Out) as soon as each request finishes (so callers can persist
        every reply immediately). Uses vLLM's enqueue + engine.step loop (what LLM.generate does internally)."""
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
        eng = self.llm.llm_engine
        while eng.has_unfinished_requests():
            for o in eng.step():
                if o.finished:
                    rid = o.request_id if o.request_id in id2i else o.request_id.rsplit("-", 1)[0]
                    yield id2i[rid], self._out(o)

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
