"""Build the peer-review dataset incl. rejected papers (login node, needs internet). Replaces the accepted-only version.

Pool/metadata: demfier/reviewertoo-iclr2025 (iclr2025_full: 11,672 submissions, decision, primary area, arXiv id).
Text: PDFs from simon123905/iclr2025_all_zips (OpenReview PDFs of papers with an accept/reject decision), read remotely
member-by-member; desk-rejected papers (not in the zips) fall back to their arXiv PDF when an arXiv id is known.
All text comes from the same extractor (PyMuPDF) so accepted/rejected papers are comparable. Main text only
(references/appendix cut).
Sample (paper App. B.3): all Oral+Spotlight, 300 Posters, 1000 Rejected, all desk-rejected with text.
The sample used in our run is stored as data/iclr2025/paper_manifest.csv (ids, decisions, areas; the full texts are NOT in the
repo because the papers are third-party material). By default this script rebuilds exactly that set from the manifest; pass
--resample to draw a new random sample instead.
Needs pymupdf + remotezip (`pip install pymupdf remotezip`, or `pip install --target DIR ...` and put DIR on PYTHONPATH).
Resumable: appends to --out, skips finished paper ids.
"""
import argparse, json, os, re, time, threading, urllib.request
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
import pymupdf
from remotezip import RemoteZip

ZIP = "https://huggingface.co/datasets/simon123905/iclr2025_all_zips/resolve/main/group_%04d.zip"
slug = lambda t: re.sub(r"[^a-z0-9]", "", t.lower())
tl = threading.local()


def zip_index(path):
    if os.path.exists(path):
        return json.load(open(path))
    idx = {}
    for g in range(9):
        for n in RemoteZip(ZIP % g).namelist():
            if n.endswith(".pdf"):
                idx.setdefault(slug(n.rsplit("/", 1)[-1][:-4]), (g, n))
    json.dump(idx, open(path, "w"))
    return idx


def with_retry(fn, tries=6):
    for i in range(tries):
        try:
            return fn()
        except Exception as e:
            if i == tries - 1:
                raise
            time.sleep(10 * (i + 1))


def zip_pdf(g, member):
    zs = tl.__dict__.setdefault("zs", {})
    if g not in zs:
        zs[g] = with_retry(lambda: RemoteZip(ZIP % g))
    return with_retry(lambda: zs[g].read(member))


def arxiv_pdf(aid):
    return with_retry(lambda: urllib.request.urlopen(urllib.request.Request(
        f"https://arxiv.org/pdf/{aid}", headers={"User-Agent": "Mozilla/5.0"}), timeout=60).read())


def main_text(pdf_bytes):
    doc = pymupdf.open(stream=pdf_bytes, filetype="pdf")
    t = "\n".join(p.get_text() for p in doc)
    t = "\n".join(l for l in t.split("\n") if not re.search(r"under review as a conference paper at iclr", l, re.I)
                  and not re.fullmatch(r"\s*\d{1,3}\s*", l))
    cuts = [m.start() for m in re.finditer(r"^\s*(references?|bibliography|appendix|[A-Z]\s+appendix|appendices)\s*$", t, re.I | re.M)
            if m.start() > 0.2 * len(t)]
    return (t[: min(cuts)] if cuts else t).strip()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/iclr2025/papers_review.jsonl")
    ap.add_argument("--pool", default="data/iclr2025/reviewertoo/iclr2025_full.parquet")
    ap.add_argument("--index", default="data/iclr2025/zip_slug_index.json")
    ap.add_argument("--n-poster", type=int, default=300)
    ap.add_argument("--n-reject", type=int, default=1000)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--manifest", default="data/iclr2025/paper_manifest.csv")
    ap.add_argument("--resample", action="store_true", help="ignore the manifest and draw a new random sample")
    ap.add_argument("--dry-run", action="store_true", help="only print the selected sample size")
    a = ap.parse_args()
    if not os.path.exists(a.pool):   # download the metadata pool (11,672 submissions) from the Hugging Face hub
        from huggingface_hub import hf_hub_download
        os.makedirs(os.path.dirname(a.pool), exist_ok=True)
        hf_hub_download("demfier/reviewertoo-iclr2025", "iclr2025_full.parquet", repo_type="dataset", local_dir=os.path.dirname(a.pool))
    df = pd.read_parquet(a.pool, columns=["paper_id", "title", "primary_area", "arxiv_id", "decision"])
    idx = zip_index(a.index)
    df["src"] = df.title.map(lambda t: idx.get(slug(t)))
    df["arx"] = df.arxiv_id.fillna("").str.len() > 0
    inz = df[df.src.notna()]
    use_manifest = os.path.exists(a.manifest) and not a.resample
    sel = df[df.paper_id.isin(set(pd.read_csv(a.manifest).paper_id))] if use_manifest else pd.concat([
        inz[inz.decision.isin(["accept (oral)", "accept (spotlight)"])],
        inz[inz.decision == "accept (poster)"].sample(a.n_poster, random_state=a.seed),
        inz[inz.decision == "reject"].sample(a.n_reject, random_state=a.seed),
        df[(df.decision == "desk reject") & df.arx],
    ])
    if a.dry_run:
        print(f"sample {len(sel)} ({sel.decision.value_counts().to_dict()}), from {'manifest' if use_manifest else 'random draw'}")
        raise SystemExit
    done = {json.loads(l)["paper_id"] for l in open(a.out)} if os.path.exists(a.out) else set()
    todo = [r for r in sel.itertuples() if r.paper_id not in done]
    print(f"sample {len(sel)} ({sel.decision.value_counts().to_dict()}), todo {len(todo)}", flush=True)
    lock = threading.Lock()

    def work(r):
        try:
            data = zip_pdf(*r.src) if r.src else arxiv_pdf(r.arxiv_id)
            txt = main_text(data)
        except Exception as e:
            print("FAIL", r.paper_id, repr(e)[:120], flush=True)
            return
        if len(txt) < 8000:
            print("SHORT", r.paper_id, len(txt), flush=True)
            return
        with lock, open(a.out, "a") as f:
            f.write(json.dumps({"paper_id": r.paper_id, "title": r.title, "decision": r.decision,
                                "type": r.decision, "primary_area": r.primary_area, "text": txt}) + "\n")

    with ThreadPoolExecutor(a.workers) as ex:
        for i, _ in enumerate(ex.map(work, todo), 1):
            if i % 50 == 0:
                print(f"{i}/{len(todo)}", flush=True)
    print("done", sum(1 for _ in open(a.out)))
