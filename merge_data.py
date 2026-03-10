"""
merge_data.py

Strategy:
  - "Multi-sample" CSVs (many rows per disease, e.g. Final_Augmented) → kept as-is,
    used as the row base.
  - "Lookup" CSVs (one row per disease, e.g. trainings, symbipredict) → collapsed to
    one symptom row per disease name, then LEFT-JOINED onto the base to add extra columns.

This preserves the thousands of training samples from Final_Augmented while
still enriching each row with symptom columns from the other sources.
"""

import re
import pandas as pd
from pathlib import Path

CLEAN_DIR  = Path("cleaned_data")
MERGED_DIR = Path("merged")
MERGED_DIR.mkdir(exist_ok=True)
OUTPUT_PATH = MERGED_DIR / "merged_dataset.csv"

def normalize_columns(cols):
    return [re.sub(r"[^\w]+", "_", str(c).strip().lower()).strip("_") for c in cols]

def normalize_name_col(s: pd.Series) -> pd.Series:
    return (s.astype(str)
             .str.strip()
             .str.lower()
             .str.replace(r"[^a-z0-9]+", "_", regex=True)
             .str.strip("_"))

def load_and_normalise(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    df.columns = normalize_columns(df.columns)
    if "name" not in df.columns:
        return None
    df["name"] = normalize_name_col(df["name"])
    name_cols   = [c for c in ["code", "name"] if c in df.columns]
    num_cols    = df.drop(columns=name_cols, errors="ignore").select_dtypes(include="number").columns.tolist()
    binary_cols = [c for c in num_cols if df[c].dropna().isin([0, 1, 0.0, 1.0]).all()]
    return df[name_cols + binary_cols]

# ── load ──────────────────────────────────────────────────────────────────────

csv_files = sorted(CLEAN_DIR.glob("*.csv"))
print(f"Found {len(csv_files)} CSV(s) in {CLEAN_DIR}")

loaded = {}
for f in csv_files:
    df = load_and_normalise(f)
    if df is None:
        print(f"  skipping {f.name} — no 'name' column")
        continue
    avg = len(df) / df["name"].nunique()
    print(f"  {f.name}: {len(df)} rows, {df['name'].nunique()} diseases, avg {avg:.1f} rows/disease")
    loaded[f.name] = df

if not loaded:
    raise RuntimeError("No valid CSVs found.")

# ── split into multi-sample bases vs lookup tables ────────────────────────────

bases   = {k: v for k, v in loaded.items() if len(v) / v["name"].nunique() > 1.5}
lookups = {k: v for k, v in loaded.items() if len(v) / v["name"].nunique() <= 1.5}

print(f"\nBase datasets (multi-sample):      {list(bases.keys())}")
print(f"Lookup datasets (one-row/disease): {list(lookups.keys())}")

# ── row base ──────────────────────────────────────────────────────────────────

if bases:
    base_df = pd.concat(bases.values(), axis=0, ignore_index=True)
else:
    print("  No multi-sample base — stacking everything.")
    base_df = pd.concat(loaded.values(), axis=0, ignore_index=True)

print(f"\nBase shape before lookup join: {base_df.shape}")

# ── enrich with lookup columns ────────────────────────────────────────────────

if lookups:
    lookup_frames = []
    for df in lookups.values():
        lk = df.drop(columns=["code"], errors="ignore").groupby("name").max()
        lookup_frames.append(lk)

    combined_lookup = lookup_frames[0]
    for lk in lookup_frames[1:]:
        combined_lookup = combined_lookup.join(lk, how="outer", rsuffix="_r")
        combined_lookup = combined_lookup.T.groupby(level=0).max().T

    code_series = base_df.pop("code") if "code" in base_df.columns else None
    base_df = base_df.set_index("name")
    base_df = base_df.join(combined_lookup, how="left", rsuffix="_lk")
    base_df = base_df.T.groupby(level=0).max().T
    base_df = base_df.reset_index()
    if code_series is not None:
        base_df.insert(0, "code", code_series.values)

# ── cleanup ───────────────────────────────────────────────────────────────────

base_df = base_df.fillna(0).infer_objects(copy=False)
base_df = base_df.drop_duplicates()

meta = [c for c in ["code", "name"] if c in base_df.columns]
syms = sorted([c for c in base_df.columns if c not in meta])
base_df = base_df[meta + syms]

base_df.to_csv(OUTPUT_PATH, index=False)

print(f"\nSaved → {OUTPUT_PATH}")
print(f"  rows            : {len(base_df)}")
print(f"  unique diseases : {base_df['name'].nunique()}")
print(f"  symptom cols    : {len(syms)}")
print(f"  avg rows/disease: {len(base_df)/base_df['name'].nunique():.1f}")
print("\nmerge_data.py done ✓")