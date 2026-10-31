"""Convert BLS CPS Table 11 (cpsaat11.xlsx: employed persons by detailed occupation, sex, race, Hispanic or Latino
ethnicity; https://www.bls.gov/cps/cpsaat11.xlsx, download in a browser: bls.gov returns 403 to scripts) into
data/hiring/bls_directions.json = stereotype directions used by the adversarial hiring persona and the bias metric.

Columns after the title: total employed, % women, % White, % Black, % Asian, % Hispanic.
Gender order = majority gender first. Ethnicity order = over-representation (group share in the occupation / group share in
all occupations); --raw-share orders by raw share instead (the paper only says "expected direction of occupational
stereotyping").  Needs openpyxl (`pip install openpyxl`).
  python scripts/prepare_bls.py data/hiring/cpsaat11.xlsx
"""
import argparse, json, os, sys
import pandas as pd
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from steering_agents.apps.hiring_data import OCC, W, B, A, H, FM, MF

ap = argparse.ArgumentParser()
ap.add_argument("xlsx")
ap.add_argument("--out", default="data/hiring/bls_directions.json")
ap.add_argument("--raw-share", action="store_true")
a = ap.parse_args()
df = pd.read_excel(a.xlsx, header=None)
num = lambda x: pd.to_numeric(x, errors="coerce")
rows = {str(r[0]).strip().lower(): [num(v) for v in r[1:7]] for r in df.itertuples(index=False) if isinstance(r[0], str)}
base = next((v for k, v in rows.items() if k.startswith("total, 16 years")), None)
out, missing = {}, []
for occ in OCC:
    v = rows.get(occ.lower())
    if v is None or any(pd.isna(x) for x in v[1:6]):
        missing.append(occ)
        continue
    shares = dict(zip((W, B, A, H), v[2:6]))
    if not a.raw_share and base is not None:
        shares = {g: s / base[2 + i] for i, (g, s) in enumerate(shares.items())}
    out[occ] = [list(FM if v[1] >= 50 else MF), sorted(shares, key=lambda g: -shares[g])]
json.dump(out, open(a.out, "w"), indent=1)
print(f"{len(out)} occupations written to {a.out}; missing/suppressed (hand-coded fallback kept): {missing}")
