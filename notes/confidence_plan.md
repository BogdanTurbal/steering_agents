# Plan: measuring agent confidence in the MAS

Goal: give every worker reply a confidence value, test whether it predicts (1) correctness and (2) opinion switching
between rounds, and only then let the orchestrator use it. Model: Qwen3.8-27B with thinking, served by vLLM (`llm.py`).

## 0. Unit of analysis and notation
- One observation = one worker reply: (run r, round t, agent i), score s_{i,t} on the app's scale {1..K}
  (coding/hiring K=5, review K=10).
- Peers' mean m_{-i,t} = mean of the other agents' scores at round t (the state agent i sees at t+1).
- Every confidence measure below is stored per reply; none is shown to other agents in Phase A.

## 1. Confidence measures (per reply)

| id | family | what | cost |
|---|---|---|---|
| C0 | free proxy | thinking length (`think_tokens`, already stored), normalized within (app, agent, round) | 0 |
| C1 | token likelihood | distribution p(k) over the K score values at the score position, after thinking | 1-2 one-token probes, prefix cached |
| C1n | token likelihood, no thinking | same distribution, thinking disabled, forced `{"score": ` prefix | 1 short prefill (not cached) |
| C2 | verbalized | `confidence` 1-5 with the hiring rubric (exists in hiring; add to coding and review personas as a private field) | ~10 output tokens |
| C2p | verbalized, probabilistic | distribution over the confidence token 1-5, read the same way as C1 | 1 one-token probe |

Dropped (2026-11-16): stability over re-samples (C3s) and over paraphrased prompts (C3p). They need 5-8 extra full
generations per probed reply and would slow the runs too much. All measures kept above come out of the main run itself.

Features from a distribution p over {1..K} (C1, C1n, C2p):
- `p_max` = max_k p(k)
- `H` = -Σ p(k) log p(k) / log K (normalized entropy; confidence = 1 - H)
- `E` = Σ k p(k) (expected score), `σ` = sqrt(Σ (k-E)² p(k))
- `p_deploy` / `p_accept` = probability mass above the app's decision threshold (§3)
- `mass` = probability on valid digits before renormalizing (sanity check: low mass means the probe hit the wrong spot)

Expected behaviour to check first: after a long reasoning trace C1 may be near one-hot (p_max ≈ 1) and carry little
signal. C1n and the gap C1n→C1 ("how much thinking settled the score") are there for that case, and that case is also
why C2 is kept as a separate measure (Yoon et al. 2025: verbalized confidence of reasoning models is better
calibrated than that of non-reasoning models).

## 2. How C1/C1n/C2p are read in vLLM (implemented: `steering_agents/confidence.py`, `Engine.stream`)
The main generation stays as it is (temperature 1.0, top_p 0.95, top_k 20, no logprobs). When a worker reply finishes:
1. Find the number in the answer (text after `</think>`, regex `"score"\s*:\s*`), map its character offset to the token
   index j (binary search on `len(decode(out_ids[:j]))`), and check that `out_ids[j]` is a digit and the boundary is exact.
2. Probe: 1-token request on `prompt_token_ids + out_ids[:j]` with `logprobs=20`. It is enqueued into the running vLLM
   engine right away, so the prefix (prompt + thinking + answer) is still in the prefix cache. Read the logprobs of "1".."9".
3. K=10 (review): Qwen tokenizes numbers digit by digit, so "10" = "1","0". A second probe on prefix + "1" gives
   q = P("0" | "1"); then p(10) = p("1")·q, p(1) = p("1")·(1-q).
4. Same at the `"confidence"` value (C2p).
5. C1n: the same messages rendered with `enable_thinking=False`, plus a forced `{"score": `, 1-token probe (+ the 1/10
   probe for review). The prompt part is shared with the main request, so it is also served from the cache.
6. The reply is yielded to the Runner (and written to `records_shard*.jsonl`) only after its probes are back. The probes
   are 1-token requests, so this adds a step or two per reply.
Logprobs are vLLM V1's default `raw_logprobs` (before temperature/top-k/top-p), so the sampling settings do not distort
them. `prompt_logprobs` is not used.

Stored per worker record (`conf`; orchestrator replies have none):
`{"score": {"p": [p(1)..p(K)], "mass": m, "written": "7"}, "score_nothink": {...}, "confidence": {...}, "cache": f}`.
Each entry is null when its number could not be located. `mass` = raw probability on the valid digits (low = the probe was
not at a number), `cache` = share of probe prompt tokens served from the prefix cache (should be close to 1).
C0 is `think_tokens`, C2 is `parsed["confidence"]`, both already in the record.

## 3. Ground truth per app ("is the agent right?")

| app | label y | agent's prediction from its reply | notes |
|---|---|---|---|
| coding | CodeQL alert-free code at round t | security reviewer: verdict s ≥ 3 (same threshold as "deployed"); probabilistic p_deploy = P(s ≥ 3) | only the security reviewer has a matching ground truth; the other 3 enter only the switching analyses |
| review | ICLR 2025 decision: accept (oral/spotlight/poster) vs reject | s ≥ 6; p_accept = P(s ≥ 6) | sensitivity: threshold matched to the sample's acceptance rate; also the meta-reviewer `decision` field |
| hiring | no true fit. (a) invariance: reference = mean round-1 base score over the 40 demographic variants of the pair; error = abs(s - ref) (b) adversary condition: the adversarial agent is wrong by construction in rounds 1-2 | continuous error instead of binary correct | (b) tests whether a biased agent shows lower confidence and whether its confidence pulls peers |

## 4. Hypotheses and tests
Clustering: replies are nested in runs and items, so all CIs use a cluster bootstrap over items (1000 resamples), and
regressions use mixed effects with a random intercept per item (and per run when trials > 1).
Primary tests: H1 and H2 for each app. The rest are secondary.

**H1 Validity: confidence predicts correctness.** For each measure: AUROC(confidence → correct). For p_deploy /
p_accept, also Brier score and ECE (10 bins) against y. Hiring: Spearman(confidence, -error). Area under the
risk-coverage curve (AURC) to compare measures. Evaluate separately for round 1 (independent opinions) and rounds 2-5
(after exposure to peers).

**H2 Switching: low-confidence agents move, confident peers pull.** Per agent, per transition t→t+1:
- switch_{i,t} = 1[s_{i,t+1} ≠ s_{i,t}], move Δ_{i,t} = s_{i,t+1} - s_{i,t}, gap g_{i,t} = m_{-i,t} - s_{i,t}.
- Logistic: switch ~ c_{i,t} + c̄_{-i,t} + |g_{i,t}| + agent + t + (1|item).
- Influence model (DeGroot / Friedkin-Johnsen weight): Δ_{i,t} = (a + b·c_{i,t} + d·c̄_{-i,t})·g_{i,t} + ε.
  Prediction: b < 0 (confident agents hold), d > 0 (confident peers pull).
- Also check whether a confidence-weighted peer mean m^w_{-i,t} = Σ_j c_j s_j / Σ_j c_j predicts s_{i,t+1} better than
  the plain mean (held-out R²). If it does, agents already react to confidence implicitly (through the reasoning text),
  even though the number is never shown.
Here c = each measure in turn (z-scored within app × agent). In hiring, a peer's C2 is never shown, so its effect can
only come through the reasoning text.

**H3 Useful switching.** Classify each switch (coding-security and review only) as wrong→right, right→wrong or neutral.
Test whether the share of wrong→right among switches falls as the switcher's confidence at t rises. This is the
"agents that are wrong will be less confident" claim in operational form.

**H4 Reasoning-model specifics.** (a) Sharpness: compare H of C1n and C1. (b) Which family discriminates better for this
reasoning model: compare AUROC of C1 vs C1n vs C2 vs C2p (paired bootstrap). (c) C0: does thinking length carry the same signal?
(d) Optional, subset: truncate the thinking at 10/30/50/70/90% and probe the score (C1 at each prefix), following Yoon
et al.'s trajectory analysis.

**H5 Confidence also homogenizes.** Over rounds, check whether mean confidence rises and its spread across agents
shrinks, and whether this happens faster than accuracy rises. This is the confidence counterpart of the existing coding
result (perceived readiness 98.5% vs 67% CodeQL pass rate). Report confidence-weighted conformity next to the paper's
conformity curve.

**H6 Confidence separates benign from malignant homogenization.** Definitions: benign = the panel homogenizes toward the
truth, malignant = it homogenizes toward a falsehood. The paper (arXiv:2610.09824) does not make this distinction and
counts all convergence as a failure. Homogenization: the panel's score variance falls from round 1 to round T,
var_T < var_1 (also per transition, as in Def. 3.2). The truth is the §3 label:
- Benign: the panel converges and the panel verdict at T (mean ≥ threshold) matches y. Hiring: the panel mean at T is
  within ±0.5 of the pair reference, i.e. it converges to the bias-free score.
- Malignant: the panel converges and the verdict at T is wrong. Coding: "deployable" while CodeQL alerts remain (the
  current main result). Review: against the ICLR decision. Hiring: away from the reference, e.g. toward the adversary's
  direction. A special case worth counting separately: an agent that was correct in round 1 ends up wrong.
- Neutral: no convergence.
Continuous version per run, with no thresholds: homogenization Hm_r = var_1 - var_T and distance to the truth at the
end, D_r = |panel mean_T - truth| (binary y: truth = the scale endpoint on the correct side of the threshold). Benign vs
malignant then becomes the sign and size of D_1 - D_T among runs with Hm > 0 (did the panel move toward or away from
the truth while it converged).

Confidence features for each run, used to predict the class:
- *Ex ante (round 1, before any exposure):*
  - confidence gap CG = mean confidence of correct agents - mean confidence of wrong agents. Prediction: CG > 0 → benign,
    CG < 0 ("confident liar") → malignant.
  - confidence of the minority-side agents compared with the majority.
  - mean and spread of panel confidence.
- *Trajectory:*
  - calibration gap per round = mean confidence - panel accuracy. Prediction: in malignant runs confidence rises while
    accuracy stays flat.
  - who moves: the confidence at t of agents that switch right→wrong vs wrong→right (links to H3).
- *Hiring `adv`:* the adversary's confidence in rounds 1-2 against the size of the bias that remains at T.
Tests: multinomial logistic regression class ~ CG + minority confidence + mean confidence + app-specific controls (item
difficulty proxy: round-1 score spread). Also Q ~ CG among runs with Hm > 0. Report AUROC of CG for benign vs malignant,
with a cluster bootstrap over items. Malignant runs can be rare in some apps (hiring `base`, review), so report class
counts first. If an app has fewer than ~30 runs in a class, the result is descriptive only. In Phase B, the question
becomes: does showing calibrated confidence move runs from malignant to benign?

## 5. Phase B: the orchestrator uses confidence (only if H1 holds for some measure)
Calibrate first: map the best measure to P(correct) with isotonic or Platt regression, fit on held-out items (ConfMAD
does the same). New conditions (`--cond`), compared with `base` on homogenization metrics and accuracy:
- `conf-shown`: the shared state lists each agent's calibrated confidence next to its score.
- `conf-orch`: only the orchestrator (generator, meta-reviewer, and the hiring panel aggregate) receives confidences,
  plus summary statistics: confidence-weighted mean, confidence spread, and a flag for "confident dissent" (an agent with
  high confidence and |g| ≥ 1).
- `conf-weighted`: the panel decision is the confidence-weighted mean (no change to prompts). This isolates the
  aggregation effect from the persuasion effect.
Risk to measure: confidence mimicry and inflation once confidence is visible (H5 under `conf-shown`).

## 6. Implementation (done 2026-11-16; tested with fakes and with Qwen3-0.6B locally, never run on a GPU or real vLLM)
One switch: `--confidence` on `scripts/run_mas.py`, or on the launcher, which passes it to every job and keeps it on
`--resume`. Without the flag every code path and prompt is exactly as before.

| file | change |
|---|---|
| `steering_agents/confidence.py` (new) | `digit_ids`, `splits_ten` (does the tokenizer write "10" as "1","0"), `locate` (token index of a written number), `probe_prefixes`, `distribution` (p(k) incl. the 1/10 split), `assemble` (the `conf` dict), `features` (p_max, 1-entropy, E, sd, p_hi, mass) |
| `steering_agents/llm.py` | `Out.conf`; `Engine.render(..., thinking=False)`; `Engine.stream(batch, probes)` enqueues the probes of §2 into the same vLLM step loop as soon as a reply finishes and yields the reply when its probes are back; `Engine._probe_prefixes` |
| `steering_agents/mas.py` | `App.probe_spec(call)` (default None); the Runner passes the specs to `stream` and writes `rec["conf"]` |
| `steering_agents/personas.py` | `with_confidence(fmt)` appends the hiring-worded `"confidence"` field to an output-format prompt |
| `steering_agents/apps/{coding,review,hiring}.py` | `confidence=False` argument; coding/review reviewers get the field in their output formats (review: also in the trailing reminder) and parse it (falls back to score-only); `probe_spec` = `{"score": K, "confidence": 5}` for workers only |
| `scripts/run_mas.py` | `--confidence` |
| `scripts/launch_mas_pipeline.py` | `--confidence` (adds it to `--extra`, stored in `launch_manifest.json`, so `--resume` keeps it); each job ends with `confidence_table.py --apps <app>` |
| `scripts/confidence_table.py` (new) | `<run>/analysis/confidence/replies_<app>.csv` + `summary_<app>.json` (one file per app: the app jobs run in parallel) |
| `tests/confidence_test.py` (new) | probe logic against a fake vLLM and a character-level tokenizer |
| `tests/fake_engine_test.py` | covers the confidence path (`CONF=1`, default) and the old path (`CONF=0`); ships its own 2-paper review fixture |
| `tests/local_vllm/vllm/__init__.py` (new) | local stand-in for the vLLM calls the engine makes (`LLM`, `enqueue`, `llm_engine.step`, `SamplingParams`), backed by transformers on MPS/CPU; for local development only |
| `tests/local_model_smoke.py` (new) | the real Engine, Runner and apps with `confidence=True` on a small Qwen through that stand-in |

Peers never see an agent's confidence: the coding history and the review discussion show only score + text (hiring
already worked this way). Orchestrator replies (code generator, meta-reviewer, author) are not probed.

`replies_<app>.csv` columns: app, run, item, cond, t, agent, score, think_tokens (C0), c2 (verbalized), `c1_*` / `c1n_*` /
`c2p_*` (p_max, conf_ent, E, sd, mass, and p_hi for the score), probe_cache, probe_mismatch, next_score, switch,
peer_mean, pred (score ≥ threshold), and the §3 labels: y and correct (coding: security reviewer only, and only after
`codeql_eval.py` has written `codeql_results.jsonl`; review: all reviewers), pair / ref / err (hiring).

## 7. Setup and local testing
### 7.1 Environment
- Python ≥ 3.10 (the code uses `str | None` annotations; the macOS system Python 3.9 cannot import `llm.py`).
- GPU jobs: `vllm==0.25.1`, `torch`, `transformers`, `numpy`, `pandas`, `matplotlib` in one env (the env the existing
  runs used). Nothing new is needed for the confidence code. `confidence_table.py` needs only `numpy` + `pandas` (no scipy).
- Model `Qwen/Qwen3.8-27B` (bf16) in `HF_HOME`, downloaded on the login node (jobs run with `HF_HUB_OFFLINE=1`).
- Data, as in the README: coding and hiring data are in the repo; review needs `python scripts/prepare_iclr_data.py`
  once (writes `data/iclr2025/papers_review.jsonl`); coding labels need the CodeQL bundle in `tools/codeql_bundle/`.
- Cluster settings are read from the environment by the launcher: `STEERING_ENV_SETUP` (e.g. `"module load anaconda3;
  conda activate steering_agents"`), `STEERING_TMP`, `HF_HOME`, `SLURM_GPU_PARTITION`, `SLURM_CPU_PARTITION`, `MAIL_USER`.

### 7.2 What was run locally (macOS, no GPU, no vLLM), all passing
```
R() { uv run --no-project -q --python 3.12 --with numpy --with pandas python "$@"; }   # any Python >= 3.10 env works
R tests/confidence_test.py                                   # prints a conf dict, then "confidence_test ok"
KEEP=/tmp/ft R tests/fake_engine_test.py                     # CONF=1: confidence path; KEEP keeps the output dir
CONF=0 R tests/fake_engine_test.py                           # old path, prompts unchanged
mkdir -p /tmp/ft/run && mv /tmp/ft/{coding,hiring,review} /tmp/ft/run/
R scripts/confidence_table.py --run-dir /tmp/ft/run          # writes replies_<app>.csv + summary_<app>.json
R scripts/launch_mas_pipeline.py --tier smoke --name conftest --confidence --dry-run   # inspect experiments/conftest_*/slurm/*.sbatch, then delete the dir
```
What these cover:
- `confidence_test.py` runs the real `Engine.stream` with `vllm` replaced by a fake module (fake `enqueue`/`step`,
  internal "<n>-<rand>" vs external "<n>" request ids as in vLLM 0.25.1). It checks:
  - a score written inside the thinking is ignored; an escaped `\"score\": 3` inside the review text is skipped;
  - truncated replies (no `</think>`) and replies without probe specs get no probes;
  - all 5 probes of a 1-10 reply are sent: score, score + "1", no-thinking score, no-thinking + "1", confidence;
  - their prefixes end exactly at `"score": ` / `"confidence": ` and at `<think></think>{"score": `;
  - p(k) and the 1/10 split are numerically correct, and `written` and `cache` are filled.
- `fake_engine_test.py`: all three apps for 3 rounds, with 10% garbage replies to exercise the retries. It checks that
  every worker record has `conf` and a parsed `confidence`, that no orchestrator record has `conf`, and that resume
  replays without generating anything.
- `confidence_table.py` on the fake run: all columns are filled, the labels join (review: from `decision`; hiring: ref/err),
  and `probe_mismatch` = 0.
- Launcher dry run: every job gets `--confidence` and a final `confidence_table.py --apps <app>` line;
  `launch_manifest.json` stores `"extra": "--confidence"`.

### 7.3 Real-model check without a GPU (Qwen3-0.6B, 2026-11-16)
vLLM 0.25.1 does not install on macOS (it depends on `pynvvideocodec`, which has Linux/Windows wheels only). So
`tests/local_vllm/` provides a small transformers-backed stand-in for the vLLM calls the engine makes, and
`tests/local_model_smoke.py` runs the real `Engine`, `Runner` and apps on it:
```
pip install torch transformers accelerate numpy pandas huggingface_hub    # Python >= 3.10, no vllm
PYTHONPATH=tests/local_vllm python tests/local_model_smoke.py --out /tmp/lms
python scripts/confidence_table.py --run-dir /tmp/lms/run
```
Setup: `Qwen/Qwen3-0.6B` (bf16, MPS on a 36 GB M-series Mac), 2 rounds, `confidence=True`. Data: hiring with 1 pair ×
1 group (base + adversary run), coding with 2 stubs, review with 1 short made-up paper. ~8 min in total.

Results (identical in a rerun with the in-repo script, same seeds):

| app | worker replies | probes found C1 / C1n / C2p | mean mass C1 / C1n / C2p | mismatches | parse retries |
|---|---|---|---|---|---|
| hiring | 16 | 1.00 / 1.00 / 1.00 | 1.000 / 1.000 / 1.000 | 0 | 0 |
| coding | 16 | 1.00 / 1.00 / 0.94 | 1.000 / 1.000 / 1.000 | 0 | 1 reviewer, 2 generator |
| review | 6 | 1.00 / 1.00 / 1.00 | 1.000 / 1.000 / 0.996 | 0 | 0 |

Coding C2p 0.94 = 1 reply where the model left out the confidence field. It was parsed as score-only (the intended
fallback), so there was nothing to probe. Examples of stored `conf` (rounded):
- coding security reviewer, wrote 3, verbalized confidence 4: C1 p(3) = 0.99; C1n p(3) = 0.39 and p(4) = 0.50;
  C2p p(4) = 0.55 and p(5) = 0.43. Thinking sharpens the score distribution, as expected in §1.
- review pragmatist, wrote 8: C1 p(8) = 0.86, p(7) = 0.08; p(10) = 0.0035 via the 1/10 split.
- hiring subject evaluator, wrote 5: C1 p(5) = 0.99; C1n p(5) = 0.93.
Tokenizer facts confirmed on the Qwen3 tokenizer:
- every digit is one token, and "10" is written as "1","0" (`splits_ten` = True);
- the space before a number is its own token;
- with thinking off, the chat template ends with an empty `<think>\n\n</think>\n\n` block, so C1n really has no thinking.

What this proves: on a real Qwen tokenizer, chat template and model output, the probes sit exactly before the number
that was written, read a distribution with almost all mass on valid digits, and run through the Runner into the records
and the table.

Still not covered (needs the cluster, see §8):
- vLLM accepting `{"prompt_token_ids": ...}` in `LLM.enqueue`, and `RequestOutput.num_cached_tokens`;
- prefix-cache behaviour of the hybrid Qwen3.8 (the stand-in has no cache, so `cache` = 0 here);
- whether Qwen3.8-27B tokenizes the same way as Qwen3 (very likely, same family);
- speed.

## 8. Handover (for the person running it on the cluster, and their coding agent)
State: everything in §6 is implemented and passes the CPU tests in §7.2. Nothing has run on a GPU. The goal of the
cluster work is to collect confidence data (C0, C1, C1n, C2, C2p) in the normal MAS runs. The analysis (§4) is done
afterwards on the CSV files, without a GPU.

### 8.1 Steps
1. Set up the env and data (§7.1). Run the two CPU tests on the login node: `python tests/confidence_test.py` and
   `python tests/fake_engine_test.py`. Both must pass before anything is submitted. (`tests/local_model_smoke.py` is for
   laptops without vLLM; on the cluster the smoke tier below is the real-model check.)
2. Smoke run (1 h, a few items per app):
   `python scripts/launch_mas_pipeline.py --tier smoke --name conf --confidence`
3. Check the smoke run (§8.2). If a check fails, see §8.3 before going further. Do not launch bigger tiers on a failed
   smoke run.
4. Medium run (one 8 h GPU job per app: coding 280 stubs, hiring 40 pairs with the adversary condition, review 300 papers):
   `python scripts/launch_mas_pipeline.py --tier medium --name conf --confidence`
   Interrupted jobs: `python scripts/launch_mas_pipeline.py --resume experiments/<run>`. Saved replies are reused, and
   `--confidence` is kept.
5. Hand back `experiments/<run>/analysis/` (`summary.json`, `quality_report.md`, figures, and `confidence/`) and
   `experiments/<run>/logs/`. The records (`<run>/<app>/records_shard*.jsonl`) hold everything else if more is needed.

### 8.2 Smoke-run checks
Files: `experiments/<run>/analysis/confidence/summary_<app>.json`, `experiments/<run>/logs/<app>_0_*.{out,err}`.
| check | expected | if not |
|---|---|---|
| job finished, `done_shard0of1.json` present | yes | read the `.err` log; §8.3 |
| `probe_ok_share` C1, C2p | ≥ 0.95 | §8.3 "probes missing" |
| `probe_ok_share` C1n | ≈ 1 (needs no located number) | §8.3 "enqueue rejects token prompts" |
| `probe_mismatch` | ≈ 0 | look at the records with mismatches (§8.3) |
| `mean_mass` C1, C2p, C1n | > 0.5, ideally > 0.9 | §8.3 "low mass" |
| `mean_probe_cache` | > 0.9 | §8.3 "cache misses" |
| review log line `WARNING: the tokenizer writes "10" as one token` | absent | review scores are not probed; §8.3 |
| gen tok/s in the `pass done` log lines | close to a run without `--confidence` | §8.3 "cache misses" |
| `failed_%` per agent in `analysis/quality_report.md` | about as before (`results/medium_20261026/quality_report.md`) | the added field confuses the format; look at raw `answer`s |

To look at single replies: `head -1 experiments/<run>/review/records_shard0of1.jsonl | python -m json.tool`, which
shows `parsed.confidence` and `conf`.

### 8.3 Known risks and what to do
- **enqueue rejects token prompts** (an error in `LLM.enqueue`/input processing about the prompt type). In
  `Engine.stream` (`steering_agents/llm.py`) replace `{"prompt_token_ids": pre[n]}` with
  `vllm.inputs.TokensPrompt(prompt_token_ids=pre[n])`.
- **`num_cached_tokens` missing** in vLLM 0.25.1: `cache` is then 0/null and the cache check cannot be done that way.
  Use vLLM's own logged prefix-cache hit rate and compare speed with a run without `--confidence` instead.
- **Cache misses** (`mean_probe_cache` low, or the run is much slower). Qwen3.8 has gated delta-net (recurrent) layers,
  and vLLM may cache their state only at block boundaries, so a probe can re-prefill a block or the whole prompt. Review
  prompts are ~100k tokens, so this matters most there. Measure first (coding/hiring prompts are short, so they may be
  fine). Ways out, in this order: (a) drop C1n for review (it is the probe least likely to share the cache);
  (b) probe only some apps. Both need a small option in `probe_spec`. Ask before changing what is measured.
- **Probes missing** (`probe_ok_share` low). `locate` returns None when the number is not right after `"score": ` at a
  token boundary. Decode one reply's `answer` and check the format (e.g. `"score": "7"` is handled; `"score": 7.5` is
  located but then dropped as a mismatch, because the written "7" differs from the parsed 7.5). Fix the regex in `confidence.locate` and add the case to `tests/confidence_test.py`.
- **Low mass**: the next token at the probe position is mostly not a digit (e.g. a space or quote token comes first).
  Print the top logprobs of one probe (add a debug print in `Engine.stream`, where `w[2][name]` is filled) and adjust the
  prefix in `confidence.locate` or `NOTHINK_PREFIX`.
- **"10" is one token**: then the 1/10 split is invalid and review scores are skipped on purpose. Fix by also reading
  the probability of the "10" token at the first position (`confidence.distribution`), and add a test.
- **Out of memory / slower decode**: probes are 1-token requests and keep no extra state, so this is unlikely. If it
  happens, lower `--max-num-seqs` via `--extra`.

### 8.4 Rules
- Do not change the model, sampling settings, prompts or tier sizes without asking. Runs must stay comparable with each
  other, and a `--confidence` run is its own baseline (not comparable to `medium_20261026`, whose prompts lacked the
  field).
- Do not re-add the stability probes (C3s/C3p): they were dropped on purpose (§1).
- Every fix to the probe code gets a case in `tests/confidence_test.py`. Run both CPU tests before resubmitting; for
  changes to prompts or parsing, also run `tests/local_model_smoke.py` on a laptop (§7.3).
- Log what was run and what was found in `notes/plan.md` (status log), with dates and job ids.

## References
- Geng et al. 2024, survey of confidence estimation and calibration in LLMs, arXiv:2311.08298
- Shorinwa et al. 2024, survey of uncertainty quantification for LLMs, arXiv:2412.05563
- Yoon et al. 2025, Reasoning Models Better Express Their Confidence (NeurIPS), arXiv:2505.14489
- Gao et al. 2024, SPUQ: perturbation-based uncertainty quantification (EACL), arXiv:2403.02509
- Kuhn, Gal, Farquhar 2023, Semantic Uncertainty (ICLR), arXiv:2302.09664
- Sheng et al. 2025, uncertainty of LLM-as-a-judge with conformal prediction over rating-token logits, arXiv:2509.18658
- Lin & Hooi 2025, ConfMAD: multi-agent debate with confidence expression, arXiv:2509.14034
- Zhu et al. 2026, Demystifying Multi-Agent Debate: confidence and diversity, arXiv:2601.19921
- Keramati et al. 2026, The Confident Liar: diagnosing MAD with log-probabilities, arXiv:2606.10296
- Farr et al. 2026, The Cost of Consensus: Malignant Epistemic Herding and Adaptive Gating in Distributed Multi-Agent
  Search, arXiv:2605.06988 (simulated Bayesian grid agents, not LLMs; MEH = low belief divergence + task failure)
- Bertalanič & Fortuna 2026, The Cost of Consensus: Isolated Self-Correction Prevails Over Unguided Homogeneous
  Multi-Agent Debate, arXiv:2605.00914 (LLM debate; vulnerability P(wrong_t | correct_t-1), recovery, oracle gap,
  verbalized 0-100 confidence found poorly calibrated)
