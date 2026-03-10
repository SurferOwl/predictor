"""
clean_data.py

Reads raw source CSVs from data/ and produces clean binary symptom
matrices in cleaned_data/.

Input files expected in data/:
  - diseases.csv              (Code, Name, Symptoms as comma-separated string)
  - Final_Augmented_dataset_Diseases_and_Symptoms_Revised.csv  (already wide)
  - symbipredict_2022.csv     (already wide)
  - trainings.csv             (already wide)

Output files written to cleaned_data/:
  - disease_symptom_matrix.csv   (from diseases.csv)
  - Final_Augmented_dataset_Diseases_and_Symptoms_Revised.csv  (copied/normalised)
  - symbipredict_2022.csv        (copied/normalised)
  - trainings.csv                (copied/normalised)
"""

import re
import pandas as pd
from pathlib import Path

RAW_DIR     = Path("data")
CLEAN_DIR   = Path("cleaned_data")
CLEAN_DIR.mkdir(exist_ok=True)


# ── helpers ───────────────────────────────────────────────────────────────────

def normalize_columns(cols):
    result = []
    for c in cols:
        c = str(c).strip().lower()
        c = re.sub(r"[^\w]+", "_", c)
        c = c.strip("_")
        result.append(c)
    return result


def collapse_duplicate_columns(df: pd.DataFrame) -> pd.DataFrame:
    """If two columns share the same name after normalisation, keep the max."""
    return df.T.groupby(level=0, sort=False).max().T


def save_normalised(src: Path, dst: Path):
    """Load a wide CSV, normalise column names, collapse dupes, save."""
    df = pd.read_csv(src)
    df.columns = normalize_columns(df.columns)
    df = collapse_duplicate_columns(df)
    df = df.fillna(0).infer_objects(copy=False)
    df.to_csv(dst, index=False)
    print(f"  ✓ {src.name} → {dst}  shape={df.shape}")


# ── 1. diseases.csv → binary matrix ──────────────────────────────────────────

diseases_src = RAW_DIR / "diseases.csv"
if diseases_src.exists():
    print("Converting diseases.csv → disease_symptom_matrix.csv ...")
    df = pd.read_csv(diseases_src)

    df["Symptoms_list"] = df["Symptoms"].apply(
        lambda x: [s.strip() for s in str(x).split(",")]
    )

    all_symptoms = sorted({
        symptom
        for symptoms in df["Symptoms_list"]
        for symptom in symptoms
        if symptom  # skip empty strings
    })

    symptom_matrix = pd.DataFrame(
    [{s: (1 if s in lst else 0) for s in all_symptoms} for lst in df["Symptoms_list"]],
    index=df.index
    )
    df = pd.concat([df[["Code", "Name"]], symptom_matrix], axis=1)

    final = df[["Code", "Name"] + all_symptoms]
    final.columns = normalize_columns(final.columns)
    out = CLEAN_DIR / "disease_symptom_matrix.csv"
    final.to_csv(out, index=False)
    print(f"  ✓ disease_symptom_matrix.csv  shape={final.shape}")
else:
    print(f"  ⚠ {diseases_src} not found, skipping.")


# ── 2. Normalise the other wide CSVs ─────────────────────────────────────────

wide_files = [
    "symbipredict_2022.csv",
    "trainings.csv",
    "training.csv",
]

print("\nNormalising wide CSVs ...")
for fname in wide_files:
    src = RAW_DIR / fname
    if src.exists():
        save_normalised(src, CLEAN_DIR / fname)
    else:
        print(f"  ⚠ {fname} not found in data/, skipping.")

print("\nstep1_clean.py done ✓")