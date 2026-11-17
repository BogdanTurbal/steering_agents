"""CPU test of the confidence probes: Engine.stream with a fake vLLM and a character-level tokenizer (no GPU, no vllm)."""
import json, math, os, sys, types
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from steering_agents import confidence as CF
from steering_agents.llm import Engine

THINK = 100000   # id of the </think> token


class Tok:
    """One token per character (digits are single tokens, as in Qwen), plus </think> as one token."""
    def encode(self, s, add_special_tokens=False):
        out, i = [], 0
        while i < len(s):
            if s.startswith("</think>", i):
                out.append(THINK); i += 8
            else:
                out.append(ord(s[i])); i += 1
        return out

    def decode(self, ids):
        return "".join("</think>" if i == THINK else chr(i) for i in ids)

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True, enable_thinking=True, **kw):
        s = "".join(f"<{m['role']}>{m['content']}" for m in messages) + "<assistant>"
        return s if enable_thinking else s + "<think></think>"


tok = Tok()
D = CF.digit_ids(tok)

# --- locate ----------------------------------------------------------------------------------------------------------
ids = tok.encode('thinking "score": 2</think>{"output": "ok", "score": 7, "confidence": 4}')
j, w = CF.locate(tok, ids, "score")
assert w == "7"
assert CF.locate(tok, tok.encode('</think>{"output": "a \\"score\\": 3 b", "score": 10}'), "score")[1] == "10"   # escaped key skipped
assert CF.splits_ten(tok)
assert tok.decode(ids[:j]).endswith('"score": ') and tok.decode(ids[j:j + 1]) == "7", "score inside the thinking was used"
assert tok.decode(ids[CF.locate(tok, ids, "confidence")[0]:][:1]) == "4"
assert CF.locate(tok, tok.encode('{"score": 7}'), "score") is None          # no </think>: truncated reply
assert CF.locate(tok, tok.encode('</think>{"output": "x"}'), "score") is None

# --- distribution: K=5 and the 1/10 split for K=10 -------------------------------------------------------------------
lg = math.log
d = CF.distribution({D["4"]: lg(0.6), D["5"]: lg(0.2), D["7"]: lg(0.1), 42: lg(0.1)}, 5, D)
assert d["p"] == [0, 0, 0, 0.75, 0.25] and abs(d["mass"] - 0.8) < 1e-9, d
d = CF.distribution({D["1"]: lg(0.5), D["8"]: lg(0.5)}, 10, D, {D["0"]: lg(0.4), 44: lg(0.6)})
assert d["p"][0] == 0.3 and d["p"][9] == 0.2 and d["p"][7] == 0.5, d
f = CF.features({"p": [0, 0, 0, 0.75, 0.25], "mass": 0.8}, 3)
assert f["p_max"] == 0.75 and abs(f["E"] - 4.25) < 1e-9 and f["p_hi"] == 1.0


# --- Engine.stream with a fake vLLM ----------------------------------------------------------------------------------
class SP:
    def __init__(self, **kw): self.__dict__.update(kw)


sys.modules["vllm"] = types.SimpleNamespace(SamplingParams=SP)
ANSWERS = {0: '<think>hmm "score": 2</think>{"output": "ok", "score": 7, "confidence": 4}',   # review, K=10
           1: '<think>x</think>{"output": "orchestrator text"}',                              # no probes requested
           2: '<think>never closed'}                                                          # truncated: no probes


class FakeLLM:
    def __init__(self):
        self.queue, self.n, self.prefixes = [], 0, []
        self.llm_engine = self

    def enqueue(self, prompts, sps, use_tqdm=False):
        ids = []
        for p, sp in zip(prompts, sps):
            self.n += 1
            ids.append(f"{self.n}-abcdefgh")
            self.queue.append((str(self.n), p, sp))
        return ids

    def has_unfinished_requests(self):
        return bool(self.queue)

    def step(self):
        cur, self.queue = self.queue, []
        outs = []
        for rid, p, sp in cur:
            if isinstance(p, dict):   # probe: put mass on "7"/"8" or, after a "1", on "0"
                pre = p["prompt_token_ids"]
                self.prefixes.append(pre)
                top = {D["0"]: lg(0.5), 9: lg(0.5)} if pre[-1] == D["1"] else {D["7"]: lg(0.6), D["8"]: lg(0.2), D["1"]: lg(0.2)}
                c = types.SimpleNamespace(text="7", token_ids=[D["7"]], finish_reason="length",
                                          logprobs=[{t: types.SimpleNamespace(logprob=v) for t, v in top.items()}])
                outs.append(types.SimpleNamespace(request_id=rid, finished=True, outputs=[c], prompt_token_ids=pre,
                                                  num_cached_tokens=len(pre) - 1))
            else:
                k = int(p.split("Q")[1].split("<")[0])
                text = ANSWERS[k]
                body = text[len("<think>"):]
                c = types.SimpleNamespace(text=body, token_ids=tok.encode(body), finish_reason="stop", logprobs=None)
                outs.append(types.SimpleNamespace(request_id=rid, finished=True, outputs=[c],
                                                  prompt_token_ids=tok.encode(p), num_cached_tokens=0))
        return outs


eng = Engine.__new__(Engine)
eng.effort, eng.max_tokens, eng.max_model_len, eng.tok, eng.llm = "medium", 1000, 100000, tok, FakeLLM()
batch = [([{"role": "user", "content": f"Q{k}"}], k) for k in ANSWERS]
spec = {"score": 10, "confidence": 5}
res = dict(eng.stream(batch, [spec, None, spec]))
assert set(res) == {0, 1, 2}
assert res[1].conf is None and res[2].conf is None and res[2].answer is None
c = res[0].conf
print(json.dumps(c))
assert c["score"] == {"p": [0.1, 0, 0, 0, 0, 0, 0.6, 0.2, 0, 0.1], "mass": 1.0, "written": "7"}, c  # p("1") split 1:10
assert c["confidence"] == {"p": [1.0, 0.0, 0.0, 0.0, 0.0], "mass": 0.2, "written": "4"}           # only "1" is in 1..5
assert c["score_nothink"] is not None and 0 < c["cache"] < 1
pre = [tok.decode(x) for x in eng.llm.prefixes]
assert any(s.endswith('"score": ') and "</think>" in s for s in pre), pre
assert any(s.endswith("<think></think>" + CF.NOTHINK_PREFIX) for s in pre), pre
assert any(s.endswith('"confidence": ') for s in pre), pre
assert len(pre) == 5, pre   # score, score|1, score_nothink, score_nothink|1, confidence
print("confidence_test ok")
