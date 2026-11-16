# Plan: measuring agent confidence in the MAS

Goal: give every worker reply confidence-like signals, test whether they predict (1) agreement with the app's proxy
label and (2) opinion switching between rounds, and only then let the orchestrator use them. Model: Qwen3.8-27B with
thinking, served by vLLM (`llm.py`).

Scope (revised 2026-11-16 after the Codex critique and the literature check in §9): this is an exploratory study of
confidence-like signals and score dynamics in one model and three simulated applications. It does not claim that the
measures are calibrated probabilities, that confidence causes switching (confidence is observed, not manipulated, until
Phase B), or that the labels are ground truth. Same-model persona agents are not independent experts.

What is already known (§9), so it is not claimed as new here:
- Low own confidence + confident peers → more flips (Herd Behavior, arXiv:2505.21588).
- Uncertainty-driven conformity is distinct from pure sycophancy (MUSE, arXiv:2605.27288).
- Many flips happen even without peers (37% in Not All Flips, arXiv:2606.00820), and harmful conformity is partly
  predictable from round-0 features (AUC 0.79, same paper).
What is new here:
- A reasoning model, and token-likelihood vs verbalized confidence compared inside a MAS.
- Realistic rating-scale apps with personas and an orchestrator.
- The link to the paper's homogenization metrics.
- Confidence itself homogenizing (H5).
- Benign vs malignant homogenization on proxy labels.

## 0. Unit of analysis and notation
- One observation = one worker reply: (run r, round t, agent i), score s_{i,t} on the app's scale {1..K}
  (coding/hiring K=5, review K=10).
- Peers' mean m_{-i,t} = mean of the other agents' scores at round t (the state agent i sees at t+1).
- Every confidence measure below is stored per reply; none is shown to other agents or the orchestrator in Phase A.
- **Independent unit = item** (coding stub, hiring pair, paper). Replies of one item share the item, the prompts and the
  orchestrator. The effective sample size is close to the number of items, not the number of replies. Every interval
  is a cluster bootstrap over items, and every result states the number of items behind it.
- Conditions: `base` (and `adv` in hiring) are the main runs. Two extra conditions exist only for this study
  (`--controls`): `nopeer` (no-peer control, H2/H3) and `rep` (extra trials, run-to-run variability). The paper's metrics
  never include them.

## 1. Confidence measures (per reply)

| id | family | what it measures (estimand) | cost |
|---|---|---|---|
| C0 | behavioural correlate | thinking length (`think_tokens`). Not confidence: it also reflects difficulty, verbosity and truncation. Normalized within (app, agent, round) | 0 |
| C1 | token likelihood | the model's next-token distribution p(k) over the K score values at the position of the written score, after its own thinking and review text. Local preference for the formatted score, not P(correct) | 1-2 one-token probes, prefix cached |
| C1n | token likelihood, separate protocol | the same distribution under a different elicitation: thinking disabled, score forced first (`{"score": `), no review written. Not "the model before thinking": it changes reasoning, context and format at once | 1-2 one-token probes, prompt cached |
| C2 | verbalized | private `confidence` 1-5 (hiring rubric; added to coding and review reviewers). Elicitation is a treatment (§1.1) | ~10 output tokens |
| C2p | verbalized, token level | next-token distribution over the confidence value 1-5. Uncertainty about the verbalized number, not a second estimate of P(correct) | 1 one-token probe |

Dropped (2026-11-16): stability over re-samples (C3s) and over paraphrased prompts (C3p). They need 5-8 extra full
generations per probed reply and would slow the runs too much. The `rep` trials (§6) give a cheap partial substitute:
round 1 is peer-free, so its trials are re-samples of the same call (test-retest of score and confidence on 10% of items).

Features from a distribution p over {1..K} (C1, C1n, C2p), in `confidence.features`:
- `p_max` = max_k p(k)
- `conf_ent` = 1 - H, with H = -Σ p(k) log p(k) / log K (normalized entropy)
- `E` = Σ k p(k) (expected value), `sd` = sqrt(Σ (k-E)² p(k))
- `p_hi` = P(value ≥ the app's threshold, §3); for the score this is P("deployable") / P("accept"), the measure that
  matches the binary proxy labels
- `mass` = probability on valid digits before renormalizing (diagnostic only)
The distributions are renormalized over the valid digits, so they are conditional on "the next token is a valid score".

Known risks for C1 (literature, §9):
- JSON-formatted logprobs saturate near 1 (VERDI, arXiv:2605.11334).
- Longer reasoning can make wrong answers more confident (arXiv:2506.18183, arXiv:2606.11211).
So C1 may be near one-hot and uninformative. That outcome is reported, not read as "the model is certain". The
opposite result for verbalized confidence of reasoning models (Yoon et al., arXiv:2505.14489) is why C2 is kept.
Verbalized and token-level calibration differ and depend on the protocol (arXiv:2605.27752, arXiv:2609.26489), so the
measures are compared, never pooled.

### 1.1 The confidence field is a treatment (manipulation check H0)
Asking for confidence in the same generation can change the answers. One paper reports lower accuracy
(arXiv:2603.25052), another higher (arXiv:2306.13063). So a `--confidence` run is its own condition.

H0 test: `scripts/compare_runs.py --a <run without --confidence> --b <run with --confidence>`, on matched runs (same
items, ids and seeds). It reports:
- the paired score difference per round (item bootstrap CI);
- the conformity / polarization / inertia curves of both runs;
- parse failures, retries and thinking length;
- the round-1 score histograms.

Reference A = the medium run of 2026-10-26 (`results/medium_20261026`, its records on the cluster). If the homogenization
curves differ beyond the CIs, the confidence results are reported as describing the confidence-elicited system only.

## 2. How C1/C1n/C2p are read in vLLM (implemented: `steering_agents/confidence.py`, `Engine.stream`)
The main generation stays as it is (temperature 1.0, top_p 0.95, top_k 20, no logprobs). When a worker reply finishes:
1. Find the number in the answer (text after `</think>`, regex `"score"\s*:\s*`), map its character offset to the token
   index j (binary search on `len(decode(out_ids[:j]))`), and check that `out_ids[j]` is a single digit and the boundary
   is exact. The number written there is stored as `written`, and the table drops the probe if it differs from the
   parsed value.
2. Probe: 1-token request on `prompt_token_ids + out_ids[:j]` with `logprobs=20`. It is enqueued into the running vLLM
   engine right away, so the prefix (prompt + thinking + answer) is still in the prefix cache. Read the logprobs of "1".."9".
3. K=10 (review): Qwen tokenizes numbers digit by digit, so "10" = "1","0" (checked at startup by `splits_ten`, and
   confirmed on the Qwen3 tokenizer). A second probe on prefix + "1" gives q = P("0" | "1"); then
   p(10) = p("1")·q and p(1) = p("1")·(1-q).
4. Same at the `"confidence"` value (C2p).
5. C1n: the same messages rendered with `enable_thinking=False`, plus a forced `{"score": `, 1-token probe (+ the 1/10
   probe for review). The prompt part is shared with the main request, so it is also served from the cache.
6. The reply is yielded to the Runner (and written to `records_shard*.jsonl`) only after its probes are back. The probes
   are 1-token requests, so this adds a step or two per reply.
Logprobs are vLLM V1's default `raw_logprobs` (before temperature/top-k/top-p), so the sampling settings do not distort
them. `prompt_logprobs` is not used.

Stored per worker record (`conf`; orchestrator replies have none):
`{"score": {"p": [p(1)..p(K)], "mass": m, "written": "7"}, "score_nothink": {...}, "confidence": {...}, "cache": f}`.
Each entry is null when its number could not be located. `cache` = share of probe prompt tokens served from the prefix
cache (should be close to 1). C0 is `think_tokens`, C2 is `parsed["confidence"]`, both already in the record.

## 3. Proxy labels per app (none of them is ground truth)

| app | label (name used in results) | agent's prediction | limits |
|---|---|---|---|
| coding | **CodeQL alert status**: the code reviewed at round t has no CodeQL security-extended alert | security reviewer: s ≥ 3 ("deployable", as in the main analysis); probabilistic: p_hi | static analysis covers a subset of issues and has false positives. Alert-free ≠ secure or deployable. Only the security reviewer is matched to it; the other three enter only the switching analyses |
| review | **historical ICLR 2025 decision**: accept (oral/spotlight/poster) vs reject | s ≥ 6; p_hi = P(s ≥ 6) | a committee outcome, not paper quality; one label per paper for all reviewers and rounds. Thresholds 5, 6, 7 are pre-specified (no prevalence-matched threshold); also the score kept continuous (rank correlation with the decision) |
| hiring | **deviation from the base reference**: ref = mean round-1 `base` score of the same pair and agent over the other demographic variants (leave-one-out, so a reply never enters its own reference); err = abs(s - ref) | continuous error | a model-made reference, not true fit or a bias-free score. Adversary condition: the adversary pushes in a pre-declared direction (its persona's stereotype ordering), so its effect is a signed shift toward that ordering, not "wrong by construction" |

The wording "correct/wrong" in the hypotheses below means "agrees / disagrees with the proxy label".

## 4. Hypotheses and tests
### 4.0 Analysis rules (fixed before the data are seen)
- **Primary analyses** (all others are exploratory and reported as such):
  - two primary measures: C2 (verbalized) and C1 `p_max`;
  - two primary tests: H1 (within-score AUROC, round 1) and H2 (own-confidence coefficient in the switch model, base
    vs nopeer);
  - for each app, Holm correction over 2 measures × 2 tests × 3 apps = 12 tests;
  - effect sizes with item-bootstrap 95% CIs are reported for everything, and p-values only for the primary family.
- **Baselines in every model:**
  - the score itself;
  - score extremity |s - midpoint| / half-range (endpoint scores can look mechanically confident);
  - round t;
  - agent role;
  - for H2/H3, the gap to the peers.

  A measure only "works" if it adds to these baselines: within-score AUROC, or a likelihood-ratio test or ΔAUROC over
  the baseline model.
- **Inference:**
  - a cluster bootstrap over items (1000 resamples) for all intervals, keeping paired comparisons paired;
  - mixed models with a random intercept per item, and rounds within a run modelled explicitly (t as a factor, lagged
    score). `confidence_table.py`'s summary numbers are descriptive only.
- **Calibration:**
  - Brier score and log loss for p_hi against the binary label;
  - reliability plots with 5 equal-mass bins;
  - ECE only as a secondary number.

  Any recalibration (isotonic or Platt) is fit by item-level cross-fitting (5 folds) and evaluated on held-out items only.
- **Power:**
  - before any test, report the number of items, labelled replies and class counts per app;
  - with fewer than ~30 items in a class the result is descriptive only;
  - coding H1 rests on one reviewer per item (≤ 280 items × 5 rounds of correlated replies).

**H1 Validity: confidence predicts agreement with the proxy label.**
- Measures: each measure's AUROC and its AUROC within each written score value (`auroc_within`).
- Baselines: the AUROC of extremity and of the score itself.
- Calibration: Brier score, log loss and reliability for p_hi.
- Hiring: Spearman(confidence, -err) and the same partial on extremity.
- Selective prediction: the area under the risk-coverage curve (AURC) to compare measures.
- Split: round 1 (peer-free) separately from rounds 2-5 (after exposure).

**H2 Switching: low-confidence agents move, confident peers pull.** Per agent, per transition t→t+1:
- Variables: switch_{i,t} = 1[s_{i,t+1} ≠ s_{i,t}], move Δ_{i,t}, gap g_{i,t} = m_{-i,t} - s_{i,t}, and
  toward_{i,t} = 1[the move goes toward m_{-i,t}].
- Models:
  - logistic: switch ~ c_{i,t} + c̄_{-i,t} + |g| + s + extremity + agent + t + (1|item);
  - logistic for toward | switch;
  - ordinal / bounded Δ instead of an unbounded linear model.
- **No-peer placebo (regression to the mean and drift).** In `nopeer` runs the four agents score the same item but
  never see each other. Fit the same models there with the placebo gap (the other nopeer agents' scores, unseen):
  - any "pull" in nopeer comes from regression to the mean or shared drift;
  - the conformity effect is the base-minus-nopeer difference (interaction cond × g, cond × c).
  - coding caveat: nopeer reviewers re-review a fixed round-1 code, while base reviewers see revised code. The coding
    contrast therefore includes the effect of the code changing, and is cleanest in hiring and review, where the item
    never changes.
  - Not All Flips (arXiv:2606.00820) found 37% of agents changing answers without peers, so this control is essential.
- Prediction (associational, since confidence is not manipulated): b < 0 (confident agents hold), d > 0 (confident peers
  pull), and both larger in base than in nopeer. Prior evidence for the same direction: Herd Behavior (arXiv:2505.21588).
- Implicit confidence: test whether a confidence-weighted peer mean m^w_{-i,t} = Σ_j c_j s_j / Σ_j c_j predicts
  s_{i,t+1} better than the plain mean (held-out items, R²). If it does, agents react to confidence through the review
  text, since the number itself is never shown.
- c = each measure in turn (z-scored within app × agent).

**H3 Useful switching (transition probabilities over all agents, not only switchers).**
- Coding-security and review: the vulnerability P(wrong_{t+1} | right_t) and the recovery P(right_{t+1} | wrong_t)
  (Bertalanič & Fortuna, arXiv:2605.00914), each as a function of the agent's confidence at t, base vs nopeer.
- Conditioning only on switches would select on the outcome, so the denominators are always all agents.
- "Agents that are wrong are less confident" in operational form: recovery falls and vulnerability rises as
  confidence rises.

**H4 Reasoning-model specifics (exploratory).**
- (a) Sharpness: compare the entropy of C1n and C1, as two elicitation protocols. This is descriptive, with no causal
  "thinking settled the score" reading.
- (b) Which measure discriminates better: AUROC of C1 vs C1n vs C2 vs C2p (paired item bootstrap).
- (c) C0: does thinking length carry signal beyond difficulty proxies?
- (d) Optional, needs extra compute: truncate the thinking at 10/30/50/70/90% and probe the score, following Yoon et
  al.'s trajectory analysis.

**H5 Confidence also homogenizes (exploratory).**
- Over rounds, check whether mean confidence rises and its spread across agents shrinks.
- Check whether this happens faster than agreement with the proxy label rises (base vs nopeer). This is the confidence
  counterpart of the coding result: perceived readiness 98.5% vs a 67% CodeQL pass rate.
- Report confidence-weighted conformity next to the paper's conformity curve.

**H6 Benign vs malignant homogenization (exploratory, continuous).** Benign = the panel homogenizes toward the proxy
label, malignant = it homogenizes away from it. The paper (arXiv:2610.09824) counts all convergence as a failure; the
precedent for "consensus + failure" is Farr et al. (arXiv:2605.06988). Per run, with no selection on convergence:
- Dispersion change ΔDisp_r = var_1 - var_T (continuous, the paper's conformity per run).
- Agreement change, at the individual level (no endpoint surrogate):
  - binary labels: ΔAgree_r = (share of agents whose verdict matches the label at T) - (same at round 1);
  - hiring: ΔErr_r = |panel mean_1 - ref| - |panel mean_T - ref|; in `adv`, the signed shift toward the adversary's
    stereotype ordering.
- Panel-level transitions, reported next to the individual ones (a correct panel can still have corrupted agents):
  right→right, wrong→right (rescued), right→wrong (corrupted), wrong→wrong (entrenched), at T vs round 1.
- Descriptive classes for figures only: benign = ΔDisp > δ and ΔAgree > 0, malignant = ΔDisp > δ and ΔAgree < 0, for
  δ = 0, 0.25·var_1 and 0.5·var_1 (threshold sensitivity). The tests use the continuous quantities.
- Predictors, strictly from round 1 and usable at decision time (no labels):
  - mean panel confidence and its spread;
  - the confidence of the minority-verdict agents vs the majority (minority = fewer agents on that side of the
    threshold);
  - a disagreement between the confidence-weighted and the plain round-1 verdict.
- Baselines: round-1 margin |panel mean_1 - threshold|, round-1 dispersion and round-1 agreement with the label.
- Model: ΔAgree ~ ΔDisp × (confidence features) + baselines, over all runs. Report the AUROC of the round-1 features
  for "malignant" vs "benign", next to Not All Flips' round-0 AUC of 0.79 for harmful conformity.
- Retrospective only (it uses labels, so the orchestrator cannot use it): the confidence gap CG = mean confidence of
  label-agreeing agents - mean confidence of disagreeing ones at round 1. "Confident liar" runs (CG < 0) vs ΔAgree.
- Trajectory (descriptive): the calibration gap per round = mean confidence - agreement with the label, and the
  confidence of agents moving right→wrong vs wrong→right.
- Hiring `adv`: the adversary's confidence in rounds 1-2 against the signed bias that remains at T.
- Base rates: report the joint distribution of (ΔDisp, ΔAgree) per app first. Malignant runs may be rare (hiring
  `base`, review).

## 5. Phase B: the orchestrator uses confidence (only if H1 holds for some measure beyond the baselines)
Phase A, the setup that exists now, collects confidence but shows it to no one: not to peers, not to the orchestrator
(code generator, meta-reviewer, author). Hiring has no orchestrator.

Phase B calibrates first. The measure that passes H1 is mapped to P(agrees with label) by isotonic or Platt regression,
cross-fitted by item, as ConfMAD does. Only a measure available at decision time can be used, so not CG. New
conditions, compared with `base` on homogenization metrics and label agreement:
- `conf-shown`: the shared state lists each agent's calibrated confidence next to its score.
- `conf-orch`: only the orchestrator receives the confidences, plus three summaries: the confidence-weighted mean, the
  confidence spread, and a "confident dissent" flag (an agent with high confidence and |g| ≥ 1).
- `conf-weighted`: the panel decision is the confidence-weighted mean, with no change to prompts. This isolates the
  aggregation effect from the persuasion effect.
- `conf-gated` (from Farr et al.'s entropy gating): an agent's update is broadcast only when its confidence changed by
  more than a threshold since its last broadcast, and peers see it weighted by confidence.

Risk to measure: confidence mimicry and inflation once confidence is visible (H5 under `conf-shown`). The literature
is mixed on whether visible confidence helps:
- it helps in ConfMAD (arXiv:2509.14034) and Zhu et al. (arXiv:2601.19921);
- it gives limited gains in arXiv:2511.07784.

## 6. Implementation (done 2026-11-16; tested with fakes and with Qwen3-0.6B locally, never run on a GPU or real vLLM)
Two switches on the launcher, both passed to every job and kept on `--resume`:
- `--confidence` (also on `scripts/run_mas.py`): the confidence field and the probes. Without it every code path and
  prompt is exactly as before.
- `--controls`: adds `--nopeer-frac 0.2 --rep-frac 0.1 --rep-trials 3` to `run_mas.py` and multiplies the job time limit
  by 1.4.
  - nopeer: a no-peer control run for the first 20% of items. In hiring only `base` is cloned, so 8 of 40 pairs × 8
    groups.
  - rep: 2 extra trials of the base condition for the first 10% of items.
  - medium tier: about +40% runs for coding and review and +20% for hiring. Nopeer review runs skip the
    orchestrator, so they are cheaper.

| file | change |
|---|---|
| `steering_agents/confidence.py` (new) | `digit_ids`, `splits_ten` (does the tokenizer write "10" as "1","0"), `locate` (token index of a written number), `probe_prefixes`, `distribution` (p(k) incl. the 1/10 split), `assemble` (the `conf` dict), `features` (p_max, 1-entropy, E, sd, p_hi, mass) |
| `steering_agents/llm.py` | `Out.conf`; `Engine.render(..., thinking=False)`; `Engine.stream(batch, probes)` enqueues the probes of §2 into the same vLLM step loop as soon as a reply finishes and yields the reply when its probes are back; `Engine._probe_prefixes` |
| `steering_agents/mas.py` | `App.probe_spec(call)` (default None); the Runner passes the specs to `stream` and writes `rec["conf"]`; `control_runs()` builds the `nopeer` / `rep` runs (ids `<main id>#nopeer`, rep = later trials); the Runner rejects duplicate run ids |
| `steering_agents/personas.py` | `with_confidence(fmt)` appends the hiring-worded `"confidence"` field to an output-format prompt |
| `steering_agents/apps/{coding,review,hiring}.py` | `confidence=False` argument; coding/review reviewers get the field in their output formats (review: also in the trailing reminder) and parse it (falls back to score-only); `probe_spec` = `{"score": K, "confidence": 5}` for workers only |
| `apps` (nopeer) | hiring: an evaluator sees only its own earlier assessments, under the header "Your earlier assessments (the other panel members' assessments are not shown)". Coding: a reviewer sees only its own earlier feedback, and the code stays the round-1 code (no generator rewrites: a generator reads every reviewer's feedback and would pass peer information on through the code). Review: a reviewer sees only its own earlier reviews; there are no meta-reviewer or author calls |
| `scripts/run_mas.py` | `--confidence`, `--nopeer-frac`, `--rep-frac`, `--rep-trials` |
| `scripts/analyze.py`, `scripts/quality_report.py` | `load()` drops `nopeer`/`rep` runs, so the paper's metrics are computed on the main runs only; the cross-trial variance (fig 7) uses base + rep |
| `scripts/code_similarity.py` | skips nopeer runs; rep runs count as extra trials of their item |
| `scripts/compare_runs.py` (new) | H0 prompt-effect check: a run without vs with `--confidence` on matched runs → `<B>/analysis/confidence/prompt_effect.json` |
| `scripts/launch_mas_pipeline.py` | `--confidence` and `--controls` (added to `--extra`, stored in `launch_manifest.json`, so `--resume` keeps them); each job ends with `confidence_table.py --apps <app>` |
| `scripts/confidence_table.py` (new) | `<run>/analysis/confidence/replies_<app>.csv` + `summary_<app>.json` (one file per app: the app jobs run in parallel) |
| `tests/confidence_test.py` (new) | probe logic against a fake vLLM and a character-level tokenizer |
| `tests/fake_engine_test.py` | covers the confidence path (`CONF=1`, default) and the old path (`CONF=0`), the nopeer and rep runs (incl. resume), and checks that a nopeer prompt contains the agent's own earlier reply and no peer's; ships its own 2-paper review fixture |
| `tests/local_vllm/vllm/__init__.py` (new) | local stand-in for the vLLM calls the engine makes (`LLM`, `enqueue`, `llm_engine.step`, `SamplingParams`), backed by transformers on MPS/CPU; for local development only |
| `tests/local_model_smoke.py` (new) | the real Engine, Runner and apps with `confidence=True` on a small Qwen through that stand-in |

Peers never see an agent's confidence: the coding history and the review discussion show only score + text (hiring
already worked this way). Orchestrator replies (code generator, meta-reviewer, author) are not probed.

`replies_<app>.csv` columns: app, run, item, cond, trial, t, agent, score, extremity, think_tokens (C0), c2 (verbalized),
`c1_*` / `c1n_*` / `c2p_*` (p_max, conf_ent, E, sd, mass, and p_hi for the score), probe_cache, probe_mismatch,
next_score, switch, peer_mean (in nopeer runs this is the placebo peer mean: peers the agent never saw), pred
(score ≥ threshold), and the §3 proxy labels:
- y and correct: coding security reviewer only, and only after `codeql_eval.py` has written `codeql_results.jsonl`;
  review: all reviewers;
- pair / ref (leave-one-out) / err: hiring.

`summary_<app>.json` (descriptive, no CIs):
- probe health;
- round-1 AUROC per measure next to the extremity baseline, and AUROC within score levels (main runs only);
- switch rate per condition (base vs nopeer);
- round-1 test-retest across the rep trials: score SD, and the rank correlation of C2 / C1 p_max / C2p E between trials.

## 7. Setup and local testing
### 7.0 From a fresh clone to the first job (cluster login node)
Run on the login node, which has internet; the compute nodes may not. `<...>` are values to fill in.
```
# 1. Code (until the PR is merged, the work is on the branch feat/confidence-measures)
git clone https://github.com/BogdanTurbal/steering_agents.git && cd steering_agents
git checkout feat/confidence-measures

# 2. Python env (any Python 3.10-3.12; vllm pulls in torch and transformers)
conda create -y -n steering_agents python=3.12 && conda activate steering_agents
pip install "vllm==0.25.1" numpy pandas matplotlib huggingface_hub pymupdf remotezip openpyxl

# 3. Cluster settings read by the launcher (put them in ~/.bashrc so later shells have them)
export HF_HOME=<big disk>/hf                          # model cache, visible from the compute nodes
export STEERING_ENV_SETUP="module load <...>; conda activate steering_agents"   # run at the start of every job
export STEERING_TMP=<node-local scratch, e.g. /tmp>   # per-job vLLM / triton / inductor caches
export SLURM_GPU_PARTITION=<gpu partition>  SLURM_CPU_PARTITION=<cpu partition>
export MAIL_USER=<optional e-mail for failure notices>

# 4. Model. llm.resolve_model looks in $HF_HOME/models--Qwen--Qwen3.8-27B/snapshots/, NOT in the default $HF_HOME/hub/,
#    so download with --cache-dir "$HF_HOME" (a plain download fails later with "not in HF_HOME").
huggingface-cli download Qwen/Qwen3.8-27B --cache-dir "$HF_HOME"
ls "$HF_HOME"/models--Qwen--Qwen3.8-27B/snapshots/   # must list one snapshot directory

# 5. Data. Coding stubs and hiring pairs are in the repo. Review needs the ICLR texts (≈1,918 papers, resumable):
python scripts/prepare_iclr_data.py                  # writes data/iclr2025/papers_review.jsonl from the manifest
wc -l data/iclr2025/papers_review.jsonl              # ≈ 1918
# CodeQL for the coding labels (version used so far: 2.27.3; check the asset name on the releases page):
mkdir -p tools/codeql_bundle && curl -L https://github.com/github/codeql-action/releases/download/codeql-bundle-v2.27.3/codeql-bundle-linux64.tar.gz | tar -xz -C tools/codeql_bundle
tools/codeql_bundle/codeql/codeql version            # must print 2.27.3

# 6. CPU tests (no GPU needed, a few seconds each)
python tests/confidence_test.py                      # ends with "confidence_test ok"
python tests/fake_engine_test.py                     # ends with "no-peer control prompts ok"

# 7. Dry run: writes the sbatch files without submitting
python scripts/launch_mas_pipeline.py --tier smoke --name conf --confidence --controls --dry-run
grep run_mas experiments/conf_*/slurm/*.sbatch       # each line must contain --confidence and --nopeer-frac
rm -r experiments/conf_*                             # remove the dry-run dir before the real launch

# 8. Submit: continue with §8.1 step 2 (smoke run)
```
Watch jobs with `squeue -u $USER`. Logs are in `experiments/<run>/logs/<app>_0_<jobid>.{out,err}`. Progress lines look
like `round 3 phase review: 1120 calls` and `pass done: ... gen tok/s ...`.

### 7.1 Environment (details)
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
- `fake_engine_test.py`: all three apps for 3 rounds, with 10% garbage replies to exercise the retries, plus nopeer
  and rep runs. It checks:
  - that every worker record has `conf` and a parsed `confidence`, and no orchestrator record has `conf`;
  - that resume replays without generating anything;
  - that control run ids never collide with main ones;
  - that a nopeer prompt contains the agent's own earlier reply and no peer's, and that nopeer review runs have no
    orchestrator calls.
- `analyze.py`, `quality_report.py` and `compare_runs.py` on the fake runs (a `CONF=0` run vs a `CONF=1` run):
  - the paper metrics count only the main runs, and the cross-trial variance includes the rep trials;
  - the comparison runs end to end, with a zero difference, because the fake's scores ignore the prompt.
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
The order matters: 2 must pass before 4, and 5 needs both runs.
1. Set up the env and data with §7.0, steps 1-7. The two CPU tests must pass, and the dry run must show the flags,
   before anything is submitted. (`tests/local_model_smoke.py` is for
   laptops without vLLM; on the cluster the smoke tier below is the real-model check.)
2. Smoke run (a few items per app, 2 rounds; 2 h limit with the controls):
   `python scripts/launch_mas_pipeline.py --tier smoke --name conf --confidence --controls`
   It prints the run directory (`experiments/conf_<date>_<time>`) and one job id per app.
3. Check the smoke run (§8.2). If a check fails, see §8.3 before going further. Do not launch bigger tiers on a failed
   smoke run.
4. Medium run with the controls (one GPU job per app, time limit 12 h instead of 8 h: coding 280 stubs, hiring 40 pairs
   with the adversary condition, review 300 papers, plus the nopeer / rep runs of §6):
   `python scripts/launch_mas_pipeline.py --tier medium --name conf --confidence --controls`
   Interrupted jobs: `python scripts/launch_mas_pipeline.py --resume experiments/<run>`. Saved replies are reused, and
   `--confidence --controls` are kept. If the cluster budget does not allow +40%, drop `--controls` and say so in the
   status log. H2's placebo test and the test-retest numbers then do not exist.
5. Prompt-effect check (H0, CPU only): find the records of the earlier medium run (`experiments/medium_20261026_*`, the
   run behind `results/medium_20261026`), then
   `python scripts/compare_runs.py --a experiments/medium_20261026_<...> --b experiments/<conf run>`.
   This writes `<conf run>/analysis/confidence/prompt_effect.json`. If those records are gone, say so. The homogenization
   curves in `results/medium_20261026/summary.json` can then be compared by eye with the new `summary.json`, but there
   is no paired score difference.
6. Hand back:
   - `experiments/<run>/analysis/` (`summary.json`, `quality_report.md`, figures, and `confidence/` incl.
     `prompt_effect.json`);
   - `experiments/<run>/logs/`;
   - `experiments/<run>/coding/codeql_results.jsonl` (the coding labels).

   The records (`<run>/<app>/records_shard*.jsonl`) hold everything else if more is needed.

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
| with `--controls`: `runs_shard*.jsonl` contains runs with `"cond": "nopeer"` and `"cond": "rep"`; `summary_<app>.json` has `switch_rate_by_cond` with a `nopeer` entry | yes | check that `--controls` reached `run_mas.py` (the sbatch file) |
| with `--controls`: `analysis/summary.json` `coverage.runs_total` per app = the main runs only (e.g. coding 4 in smoke) | yes | the controls leaked into the paper metrics; check `analyze.load` |

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
  other. A `--confidence` run is its own condition: compare it with `medium_20261026` only through the H0 check (§1.1).
- The analysis rules in §4.0 (primary measures, baselines, item-level inference, proxy-label wording) are fixed. Do not
  pick the "best" measure after looking at the results and then report it as primary.
- Do not re-add the stability probes (C3s/C3p): they were dropped on purpose (§1).
- Every fix to the probe code gets a case in `tests/confidence_test.py`. Run both CPU tests before resubmitting; for
  changes to prompts or parsing, also run `tests/local_model_smoke.py` on a laptop (§7.3).
- Log what was run and what was found in `notes/plan.md` (status log), with dates and job ids.

## 9. Critique and literature check (2026-11-16)
The approach was reviewed by Codex (a methodology critique of this plan and the code), and the literature was searched
again. Everything below has been applied to §0-§8; this section records what was found and why things changed.

### 9.1 Codex critique and what was done

| # | point | change |
|---|---|---|
| 1 | The labels are proxies, not truth: CodeQL alert-free ≠ secure or deployable; an ICLR decision is a committee outcome, not one reviewer's correctness; the hiring reference is a model-made average that also contained the scored reply | §3 renamed them (CodeQL alert status, historical decision, deviation from the base reference); "correct" now means "agrees with the proxy". The hiring reference is leave-one-out (`confidence_table.py`). Review uses pre-specified thresholds 5/6/7 instead of a prevalence-matched one, plus a continuous rank correlation |
| 2 | The confidence field is a treatment and may change scores, length, parsing and convergence | H0 manipulation check (§1.1), `scripts/compare_runs.py` against the medium run of 2026-10-26 on matched items and seeds |
| 3 | Score extremity confound: endpoint scores look confident mechanically; AUROC may measure scale use | `extremity` column. Baselines (score, extremity) are required in every model, and AUROC is also reported within each score level (`auroc_within`) |
| 4 | The effective sample size is the number of items, not replies; too many measures × scalarizations × apps × rounds | §4.0: item = unit, cluster bootstrap over items, mixed models with item intercepts and an explicit round structure, 2 primary measures × 2 primary tests with Holm correction, everything else exploratory; counts reported first |
| 5 | Switching: regression to the mean, bounded moves; H2 is associational; H3 conditioned on switching | no-peer placebo control (a second Codex review found that a coding control with code rewrites still leaked peer feedback through the generator, so coding nopeer keeps the round-1 code) (`nopeer` runs; the conformity effect = base - nopeer), ordinal/bounded move models, H2 worded as associational, H3 as transition probabilities over all agents |
| 6 | H6: the var_T < var_1 threshold is fragile, the endpoint-as-truth distance is arbitrary, selecting Hm > 0 biases the comparison, a panel verdict hides individual harm, CG uses labels and cannot be used at decision time | H6 is continuous (ΔDisp, ΔAgree at the agent level), over all runs, with threshold sensitivity for figures only. Panel transitions are reported next to individual ones. Predictors come only from round 1 and need no labels; CG is kept as retrospective only |
| 7 | C1n is a different elicitation protocol, not "the model before thinking" | §1 table and H4(a) reworded; no causal "thinking settled it" reading |
| 8 | With one trial per item there is no estimate of run-to-run variability | `rep` runs: 2 extra trials for 10% of items (round-1 test-retest of score and confidence; trial variance in fig 7) |
| 9 | Calibration: ECE with 10 bins is unstable; calibrate out of sample | Brier, log loss and 5-bin equal-mass reliability; ECE secondary; recalibration cross-fitted by item |
| 10 | The estimand of each measure is unclear | §1 table states it per measure (C0 a behavioural correlate; C1/C2p local token preference; C2 elicited statement) |
| 11 | Scope: one model, same-model personas, no independent experts | the scope paragraph at the top |

Codex cited arXiv:2605.27752 as "Same Answer, Different Confidence". The paper's title is "Asking Is Not Enough:
Protocol Sensitivity in LLM Confidence Calibration".

### 9.2 Literature: what it means for this plan
Findings from search results and abstracts. The full texts were not read unless noted, so check the numbers before
citing them.

**Directly overlapping (position against these):**
- *Not All Flips Are Conformity* (Hao et al., arXiv:2606.00820). Flip rate mixes three things: spontaneous
  instability, stance-driven conformity and reasoning-driven persuasion, separated with counterfactual conditions.
  - 37% of agent-question pairs changed their answer with no peer information at all (GPT-4o, MMLU-Pro). Among the
    stable ones, ~29% adopted a peer's answer.
  - Strict conformity was mostly harmful.
  - Harmful conformity was predictable from round-0 features (AUC 0.79), and a targeted intervention cut it by 13.6 pp.
  - What it changes here: the no-peer control and the H6 comparison.
- *Herd Behavior in LLM MAS* (arXiv:2505.21588): flips were most frequent when an agent's self-confidence was low and
  its peers looked confident; placing disagreeing opinions first amplified conformity. This is H2's prediction, already
  reported, so H2 here is a replication in a reasoning model with rating scales, token-level confidence and a no-peer
  control.
- *It's Not Always Sycophancy (MUSE)* (Guo et al., arXiv:2605.27288). It separates pure sycophancy (yielding while
  certain) from uncertainty-driven conformity, using sampling entropy as the uncertainty measure. It finds that existing
  metrics overstate pure sycophancy. This is the framing for H2: low C1/C2 + switch = uncertainty-driven; high + switch
  = pure conformity.
- *The Cost of Consensus* (Bertalanič & Fortuna, arXiv:2605.00914): debate is no better than isolated self-correction;
  verbalized confidence is overconfident and weakly predictive. Transition metrics borrowed for H3.
- *The Cost of Consensus* (Farr et al., arXiv:2605.06988): malignant epistemic herding (simulated agents, not LLMs);
  entropy gating gave Phase B's `conf-gated`.

**Measure validity (C1, C2):**
- C1 may saturate:
  - VERDI (arXiv:2605.11334): token logprobs saturate near 1.0 with structured JSON output.
  - *Reasoning about Uncertainty* (arXiv:2506.18183): reasoning models are overconfident on wrong answers, more so with
    more reasoning effort.
  - *Calibration Drift Under Reasoning* (arXiv:2606.11211): calibration error first falls, then rises with the
    chain-of-thought budget.
  - Opposite finding: Yoon et al. (arXiv:2505.14489), where verbalized calibration improves as the chain of thought
    unfolds.
- Verbalized vs logprob:
  - Hsiao (arXiv:2609.10996): for post-2025 judge models verbalized confidence beats logprob G-Eval.
  - arXiv:2605.27752 and arXiv:2609.26489: the two are not interchangeable and depend on the protocol (which token,
    which context).
  - Linear probes beat both (arXiv:2512.22245; would need hidden states, not planned).
- Eliciting confidence changes the task: arXiv:2603.25052 reports lower accuracy when confidence is produced in the same
  generation; Xiong et al. (arXiv:2306.13063) report higher accuracy with CoT-style confidence prompts. Both support H0.

**Herding and cascades (context for H5/H6):**
- arXiv:2610.30769 (Hashemi et al.): LLM populations herd more than humans and their cascades self-correct less.
- arXiv:2607.03695: the context width each agent sees switches a group between herding and pooling.
- arXiv:2508.18321: losses from correct→wrong flips outweigh gains from wrong→correct.
- arXiv:2411.01271: a Bayesian model showing LLM agents form information cascades in finite time.

**Peer review priors (review app):**
- AgentReview (arXiv:2406.12708): the rating SD fell 27.2% after reviewer-AC discussion, and one biased reviewer
  changed 37.1% of decisions.
- Baumann (arXiv:2605.03202): AI reviewers agree with each other far more than humans do, and averaged AI scores
  predict ICLR 2026 decisions worse than averaged human scores. This supports treating the decision as a weak proxy.
- ReviewerToo (arXiv:2510.08867), whose ICLR 2025 data this repo uses: about 70% accept/reject accuracy for single
  personas.

**What is still new here:**
- a reasoning model with token-level (C1, C1n, C2p) and verbalized (C2) confidence measured side by side inside a MAS;
- rating-scale apps with personas and an orchestrator, tied to the paper's conformity / polarization / inertia;
- confidence homogenization (H5);
- benign vs malignant homogenization on proxy labels with deployable round-1 predictors (H6);
- a no-peer placebo for the switching models.

The claim "confidence predicts switching" is not new on its own.

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
- Hao et al. 2026, Not All Flips Are Conformity: Decomposing Stance Convergence in Multi-Agent LLM Debate,
  arXiv:2606.00820
- Herd Behavior: Investigating Peer Influence in LLM-based Multi-Agent Systems, arXiv:2505.21588
- Guo et al. 2026, It's Not Always Sycophancy: Measuring LLM Conformity as a Function of Epistemic Uncertainty (MUSE),
  arXiv:2605.27288
- VERDI: Single-Call Confidence Estimation (logprob saturation with JSON output), arXiv:2605.11334
- Reasoning about Uncertainty: Do Reasoning Models Know When They Don't Know?, arXiv:2506.18183
- Calibration Drift Under Reasoning: How Chain-of-Thought Budgets Induce Overconfidence, arXiv:2606.11211
- Hsiao 2026, Rethinking Verbalized Confidence for LLM-as-a-Judge, arXiv:2609.10996
- Asking Is Not Enough: Protocol Sensitivity in LLM Confidence Calibration, arXiv:2605.27752
- Calibration as a First-Class Criterion in LLM Evaluation, arXiv:2609.26489
- Calibrating LLM Judges: Linear Probes for Fast and Reliable Uncertainty, arXiv:2512.22245
- Closing the Confidence-Faithfulness Gap in Large Language Models, arXiv:2603.25052
- Xiong et al. 2024, Can LLMs Express Their Uncertainty? (ICLR), arXiv:2306.13063
- Tian et al. 2023, Just Ask for Calibration (EMNLP), arXiv:2305.14975
- Hashemi et al. 2026, LLM Multi-Agent Systems Are Vulnerable to Herding and Runaway Cascades, arXiv:2610.30769
- Social Networks of LLM Agents, arXiv:2607.03695
- LLMs Can't Handle Peer Pressure: Crumbling under Multi-Agent Social Interactions, arXiv:2508.18321
- Interacting Large Language Model Agents: Interpretable Models and Social Learning, arXiv:2411.01271
- Can LLM Agents Really Debate? A Controlled Study of Multi-Agent Debate in Logical Reasoning, arXiv:2511.07784
- Jin et al. 2024, AgentReview: Exploring Peer Review Dynamics with LLM Agents (EMNLP), arXiv:2406.12708
- Baumann 2026, Stop Automating Peer Review Without Rigorous Evaluation, arXiv:2605.03202
- ReviewerToo: Should AI Join The Program Committee?, arXiv:2510.08867
- CONSENSAGENT (Findings of ACL 2025), https://aclanthology.org/2025.findings-acl.1141/
