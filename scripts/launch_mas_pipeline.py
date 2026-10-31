"""Login-node launcher: sbatch chain for the homogenization MAS experiments.

  python scripts/launch_mas_pipeline.py --tier smoke --dry-run
  python scripts/launch_mas_pipeline.py --tier pilot --name pilot1

Jobs: [build_hiring GPU] -> hiring shards (GPU) ; coding shards (GPU) -> codeql (CPU) ; review shards (GPU).
Writes <run>/slurm/*.sbatch, <run>/logs, <run>/launch_manifest.json. Tier sizes: see TIERS.
"""
import argparse, json, os, shutil, subprocess, time

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
# (n_items, n_trials, shards, rounds, extra args) per app. hiring n_items = job-description/resume pairs (x8 groups x variants).
TIERS = {
    "smoke": dict(rounds=2, hours=1, sim_prompts=2, build="--max-occ 3 --per-occ 2",
                  coding=(4, 1, 1), hiring=(1, 1, 1, "--variants 1 --adversary"), review=(2, 1, 1)),
    "pilot": dict(rounds=5, hours=12, sim_prompts=20, build="",
                  coding=(40, 1, 1), hiring=(10, 1, 1, "--variants 1 --adversary"), review=(60, 1, 1)),
    # all tiers: 1 trial (user request 2026-10-26); shards sized for ~6 h each by scripts/estimate_cost.py
    "medium": dict(rounds=5, hours=8, sim_prompts=60, build="",   # 1 job per application, 8 h each (user request 2026-10-26)
                   coding=(280, 1, 1), hiring=(40, 1, 1, "--variants 1 --adversary"), review=(300, 1, 1)),
    "full": dict(rounds=5, hours=48,   # 1 trial: ~510 GPU-h (hiring ~290, review ~215), see estimate_cost.py
                  sim_prompts=100, build="",
                 coding=(280, 1, 2), hiring=(195, 1, 20, "--variants 5 --adversary"), review=(1950, 1, 30)),
}

# Cluster-specific settings come from the environment (nothing machine- or user-specific is hard-coded):
#   STEERING_ENV_SETUP   shell snippet run on the compute node before the job (modules, conda activate ...), e.g.
#                        "module load anaconda3; conda activate steering_agents"
#   STEERING_TMP         scratch dir for per-job vLLM/triton/inductor caches (default /tmp)
#   HF_HOME              Hugging Face cache with the model (default ~/.cache/huggingface)
#   SLURM_GPU_PARTITION / SLURM_CPU_PARTITION   partitions (default gpu / cpu)
#   MAIL_USER            optional e-mail for SLURM failure notices
ENV = dict(setup=os.environ.get("STEERING_ENV_SETUP", "conda activate steering_agents"),
           tmp_root=os.environ.get("STEERING_TMP", "/tmp"),
           hf_home=os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface")),
           gpu_part=os.environ.get("SLURM_GPU_PARTITION", "gpu"), cpu_part=os.environ.get("SLURM_CPU_PARTITION", "cpu"),
           mail=os.environ.get("MAIL_USER", ""))

HEAD = """#!/bin/bash
#SBATCH --job-name={name} --nodes=1 --ntasks-per-node=1 --cpus-per-task={cpus} --mem={mem} --time={time}:00:00 --partition={part}
{gres}#SBATCH --output={run}/logs/{name}_%j.out --error={run}/logs/{name}_%j.err
{mail}cd {root}
{setup}
CACHE_ROOT={tmp_root}/steering_${{SLURM_JOB_ID:-local}}
export TMPDIR="$CACHE_ROOT/tmp" VLLM_CACHE_ROOT="$CACHE_ROOT/vllm" TORCHINDUCTOR_CACHE_DIR="$CACHE_ROOT/torch_inductor" TRITON_CACHE_DIR="$CACHE_ROOT/triton" XDG_CACHE_HOME="$CACHE_ROOT/xdg"
mkdir -p "$TMPDIR" "$VLLM_CACHE_ROOT" "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR" "$XDG_CACHE_HOME"
export HF_HOME={hf_home} HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
export VLLM_USE_FLASHINFER_SAMPLER=0   # FlashInfer sampler JIT fails on compute nodes (no curand.h)
"""


def sbatch(run, name, body, dry, dep=None, gpu=True, hours=12, mem="128G", cpus=8, kind="afterok"):
    mail = f"#SBATCH --mail-type=FAIL --mail-user={ENV['mail']}\n" if ENV["mail"] else ""
    txt = HEAD.format(name=name, run=run, root=ROOT, cpus=cpus, mem=mem, time=hours, mail=mail, setup=ENV["setup"],
                      tmp_root=ENV["tmp_root"], hf_home=ENV["hf_home"],
                      part=ENV["gpu_part"] if gpu else ENV["cpu_part"], gres="#SBATCH --gres=gpu:1\n" if gpu else "") + body + "\n"
    path = f"{run}/slurm/{name}.sbatch"
    open(path, "w").write(txt)
    if dry:
        print("DRY", path, "dep", dep)
        return f"dry-{name}"
    cmd = ["sbatch", "--parsable"] + ([f"--dependency={kind}:{dep}"] if dep else []) + [path]
    jid = subprocess.check_output(cmd, text=True).strip()
    print("submitted", name, jid, "dep", dep)
    return jid


REPO_PAIRS = os.path.join(ROOT, "data", "hiring", "pairs.jsonl")   # job description / resume pairs used in our run


def job_body(run, app, T, n_items, n_trials, shards, i, extra, hours):
    """One GPU job = [data build] -> run_mas (time-boxed) -> [CodeQL / similarity] -> analysis. Everything runs on the same
    allocation (no separate CPU jobs). run_mas is wrapped in `timeout` so the post-steps still get their time slot; records
    are saved per reply, so a timed-out run_mas loses nothing (resume with --resume)."""
    total = int(hours * 60)
    reserve = {"coding": 70, "hiring": 25, "review": 25}[app]     # minutes kept for the steps after run_mas
    build = 25 if app == "hiring" else 0                            # minutes allowed for building the hiring dataset
    run_min = total - reserve - build
    lines = []
    data = ""
    if app == "hiring":
        pairs = f"{run}/hiring/pairs.jsonl"
        data = f"--data {pairs}"
        if os.path.exists(REPO_PAIRS) and not os.path.exists(pairs):   # reuse the shipped dataset: the build step is then skipped
            os.makedirs(f"{run}/hiring", exist_ok=True)
            shutil.copy(REPO_PAIRS, pairs)
        lines.append(f"mkdir -p {run}/hiring; [ -f {pairs} ] || timeout -s TERM {build}m python scripts/build_hiring_data.py --out {pairs} {T['build']}")
    lines.append(f"timeout -s TERM {run_min}m python scripts/run_mas.py --app {app} --out-dir {run}/{app} --n-items {n_items} "
                 f"--n-trials {n_trials} --rounds {T['rounds']} --shard {i}/{shards} --wave 1000 {data} {extra}")
    if app == "coding":
        if n_trials >= 2:   # the similarity judge compares trials of the same prompt (GPU, needs the model again)
            lines.append(f"timeout -s TERM 25m python scripts/code_similarity.py --run-dir {run}/coding --n-prompts {T['sim_prompts']} {extra}")
        lines.append(f"timeout -s TERM 35m python scripts/codeql_eval.py --run-dir {run}/coding --workers 4")
    lines.append(f"python scripts/analyze.py --run-dir {run} --partial; python scripts/quality_report.py --run-dir {run}")
    return "\n".join(lines)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tier", default="smoke", choices=list(TIERS))
    ap.add_argument("--name", default=None)
    ap.add_argument("--apps", default="coding,hiring,review")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--extra", default="", help="extra args for run_mas.py (e.g. --effort low)")
    ap.add_argument("--resume", default=None, help="existing run dir: resubmit only shards without a done marker (replies already "
                    "saved in records_shard*.jsonl are reused)")
    a = ap.parse_args()
    if a.resume:   # take tier/args from the original launch
        old = json.load(open(os.path.join(a.resume, "launch_manifest.json")))
        a.tier, a.extra, a.apps = old["tier"], old["args"]["extra"], old["args"]["apps"]
    T = TIERS[a.tier]
    run = os.path.abspath(a.resume) if a.resume else os.path.join(ROOT, "experiments", f"{a.name or a.tier}_{time.strftime('%Y%m%d_%H%M%S')}")
    os.makedirs(f"{run}/slurm", exist_ok=True)
    os.makedirs(f"{run}/logs", exist_ok=True)
    jobs = {}
    for app in a.apps.split(","):
        n_items, n_trials, shards, *rest = T[app]
        extra = (rest[0] if rest else "") + " " + a.extra
        ids = []
        for i in range(shards):
            if os.path.exists(f"{run}/{app}/done_shard{i}of{shards}.json"):
                continue   # finished shard (resume)
            body = job_body(run, app, T, n_items, n_trials, shards, i, extra, T["hours"])
            ids.append(sbatch(run, f"{app}_{i}", body, a.dry_run, hours=T["hours"], cpus=8))
        jobs[app] = ids
    mf = f"{run}/launch_manifest.json"
    hist = json.load(open(mf)).get("launches", []) if os.path.exists(mf) else []
    json.dump({"tier": a.tier, "run": run, "jobs": jobs, "args": vars(a), "launches": hist + [{"time": time.ctime(), "jobs": jobs}]},
              open(mf, "w"), indent=1)
    print(run)
