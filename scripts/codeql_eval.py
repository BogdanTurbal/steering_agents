"""Run CodeQL (python-/cpp-security-extended) on every code the coding MAS produced (round 1..T code of each run).

  python scripts/codeql_eval.py --run-dir experiments/<run>/coding --workers 16
Output: <run-dir>/codeql_results.jsonl with {run, round, lang, rules: [...], n_alerts}.
Files are batched (--batch per CodeQL database), one database per batch per language. Resumable by batch.
"""
import argparse, hashlib, json, os, shutil, subprocess, sys
from concurrent.futures import ThreadPoolExecutor

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
from steering_agents.codes import collect
CODEQL = os.path.join(ROOT, "tools", "codeql_bundle", "codeql", "codeql")
SUITE = {"py": ("python", "codeql/python-queries:codeql-suites/python-security-extended.qls", "py"),
         "c": ("cpp", "codeql/cpp-queries:codeql-suites/cpp-security-extended.qls", "cpp")}


def run_batch(job):
    bdir, lang, items = job
    name, suite, ext = SUITE[lang]
    marker = os.path.join(bdir, "result.json")
    if os.path.exists(marker):
        return json.load(open(marker))
    src = os.path.join(bdir, "src")
    shutil.rmtree(bdir, ignore_errors=True)
    os.makedirs(src)
    names = {}
    for run, rnd, code in items:
        fn = f"f{hashlib.md5(f'{run}|{rnd}'.encode()).hexdigest()[:12]}.{ext}"
        open(os.path.join(src, fn), "w").write(code)
        names[fn] = (run, rnd)
    cmd = [CODEQL, "database", "create", os.path.join(bdir, "db"), f"--language={name}", f"--source-root={src}", "--threads=2", "--overwrite"]
    if name == "cpp":
        open(os.path.join(bdir, "build.sh"), "w").write(
            f"cd {src}\nfor f in *.cpp; do g++ -c -w -fpermissive -o /dev/null \"$f\" 2>/dev/null || true; done\n")
        cmd += [f"--command=bash {bdir}/build.sh"]
    subprocess.run(cmd, check=True, capture_output=True)
    sarif = os.path.join(bdir, "out.sarif")
    subprocess.run([CODEQL, "database", "analyze", os.path.join(bdir, "db"), suite, "--format=sarif-latest",
                    f"--output={sarif}", "--threads=2", "--rerun"], check=True, capture_output=True)
    alerts = {}
    for res in json.load(open(sarif))["runs"][0]["results"]:
        uri = res["locations"][0]["physicalLocation"]["artifactLocation"]["uri"]
        alerts.setdefault(os.path.basename(uri), []).append(res["ruleId"])
    rows = [{"run": run, "round": rnd, "lang": lang, "rules": sorted(set(alerts.get(fn, []))),
             "n_alerts": len(alerts.get(fn, []))} for fn, (run, rnd) in names.items()]
    json.dump(rows, open(marker, "w"))
    shutil.rmtree(os.path.join(bdir, "db"), ignore_errors=True)
    return rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--batch", type=int, default=250)
    ap.add_argument("--work-dir", default=None)
    a = ap.parse_args()
    work = a.work_dir or os.path.join(a.run_dir, "codeql_work")
    items = collect(a.run_dir)
    jobs = []
    for lang in SUITE:
        its = [x for x in items if x[0].startswith(lang + "-")]
        for i in range(0, len(its), a.batch):
            jobs.append((os.path.join(work, f"{lang}_{i // a.batch:04d}"), lang, its[i:i + a.batch]))
    print(f"{len(items)} codes, {len(jobs)} batches", flush=True)
    rows = []
    with ThreadPoolExecutor(a.workers) as ex:
        for n, r in enumerate(ex.map(run_batch, jobs), 1):
            rows += r
            print(f"batch {n}/{len(jobs)} done", flush=True)
    with open(os.path.join(a.run_dir, "codeql_results.jsonl"), "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print("wrote", len(rows))
