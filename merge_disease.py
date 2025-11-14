import pandas as pd
from pathlib import Path
import re

DATA_DIR = Path("data")


def normalize_columns(cols):
    """
    - cast to str
    - strip spaces
    - lowercase
    - replace non-word chars with '_'
    - strip leading/trailing '_'
    """
    normalized = []
    for c in cols:
        c_new = str(c).strip().lower()
        c_new = re.sub(r"[^\w]+", "_", c_new)
        c_new = c_new.strip("_")
        normalized.append(c_new)
    return normalized


def load_clean_and_collapse_dupes(filename: str) -> pd.DataFrame:
    """
    Load CSV, normalize column names, then collapse duplicate columns
    by taking the max across them (so 1 wins over 0).
    Result: one single column per symptom name, e.g. just 'cough'.
    """
    df = pd.read_csv(DATA_DIR / filename)

    # normalize col names
    df.columns = normalize_columns(df.columns)

    # collapse duplicate columns (same name) by max
    # transpose -> groupby column name -> max -> transpose back
    df = df.T.groupby(level=0, sort=False).max().T

    return df


# 1. Load three tables
d_symptom = load_clean_and_collapse_dupes("disease_symptom_matrix.csv")
d_symbipredict = load_clean_and_collapse_dupes("symbipredict_2022.csv")
d_trainings = load_clean_and_collapse_dupes("trainings.csv")

print("disease_symptom_matrix columns (sample):", list(d_symptom.columns)[:20], "...")
print("symbipredict_2022 columns (sample):", list(d_symbipredict.columns)[:20], "...")
print("trainings columns (sample):", list(d_trainings.columns)[:20], "...")


# 2. Use 'name' as merge key (must exist in all 3)
if not all("name" in df.columns for df in [d_symptom, d_symbipredict, d_trainings]):
    raise ValueError("Column 'name' must exist in all three CSVs to merge on it.")

key_col = "name"

# 3. Outer merges to keep everything
merged = d_symptom.merge(
    d_symbipredict,
    how="outer",
    on=key_col,
    suffixes=("", "_symbipredict"),
)

merged = merged.merge(
    d_trainings,
    how="outer",
    on=key_col,
    suffixes=("", "_trainings"),
)

merged.columns = [
    re.sub(r"_(symbipredict|trainings)$", "", c) for c in merged.columns
]

merged = merged.T.groupby(level=0, sort=False).max().T

# 4. Replace NaN / nulls with zero
merged = merged.fillna(0)

merged = merged.drop_duplicates()

# 5. Save final merged dataset
output_path = DATA_DIR / "merged_dataset.csv"
merged.to_csv(output_path, index=False)

print(f"Merged dataset saved to: {output_path}")
print("Final shape:", merged.shape)
