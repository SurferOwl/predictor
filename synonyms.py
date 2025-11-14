import pandas as pd
import pickle
from pathlib import Path
from sentence_transformers import SentenceTransformer, util
import torch

DATA_DIR = Path("data")
INPUT_CSV = DATA_DIR / "merged_dataset.csv"
OUTPUT_CSV = DATA_DIR / "merged_dataset_synonyms.csv"
OUTPUT_SYMPTOM_LIST = DATA_DIR / "symptom_list_merged.pkl"
REPORT_TXT = DATA_DIR / "symptom_merge_report.txt"

# -----------------------------
# 1. LOAD DATA
# -----------------------------
df = pd.read_csv(INPUT_CSV)

# Non-symptom columns (adjust if needed)
NON_SYMPTOM_COLS = [c for c in ["code", "name"] if c in df.columns]

symptom_cols = [c for c in df.columns if c not in NON_SYMPTOM_COLS]
orig_symptom_cols = symptom_cols.copy()  # IMPORTANT: keep immutable for indices

print(f"Total symptom columns: {len(orig_symptom_cols)}")

# -----------------------------
# 2. EMBED SYMPTOM NAMES
# -----------------------------
symptom_texts = [c.replace("_", " ") for c in orig_symptom_cols]

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

embeddings = model.encode(
    symptom_texts,
    convert_to_tensor=True,
    normalize_embeddings=True
)

# -----------------------------
# 3. FIND SIMILAR COLUMNS (NLP)
# -----------------------------
SIM_THRESHOLD = 0.9  # you can tune this

sim_matrix = util.cos_sim(embeddings, embeddings)  # (n, n)
n = len(orig_symptom_cols)

# Disjoint-set (union-find) to build groups
parent = list(range(n))

def find(i):
    while parent[i] != i:
        parent[i] = parent[parent[i]]
        i = parent[i]
    return i

def union(i, j):
    ri, rj = find(i), find(j)
    if ri != rj:
        parent[rj] = ri

# Union similar pairs
for i in range(n):
    for j in range(i + 1, n):
        score = float(sim_matrix[i, j].item())
        if score >= SIM_THRESHOLD:
            union(i, j)

# Collect indices by root
clusters = {}
for i in range(n):
    r = find(i)
    clusters.setdefault(r, []).append(i)

groups = [indices for indices in clusters.values() if len(indices) > 1]

print(f"Found {len(groups)} synonym groups (size >= 2) with threshold {SIM_THRESHOLD}.")

# -----------------------------
# 4. MERGE GROUPS INTO ONE COLUMN
# -----------------------------
merge_report_lines = []

for g_idx, indices in enumerate(groups, start=1):
    # Use orig_symptom_cols for stable indexing
    cols = [orig_symptom_cols[i] for i in indices]

    # Only keep columns that actually still exist in df (in case of re-runs)
    cols = [c for c in cols if c in df.columns]
    if len(cols) <= 1:
        continue  # nothing to merge

    # Canonical = shortest name
    canonical = min(cols, key=len)

    merge_report_lines.append(f"Group {g_idx}:")
    merge_report_lines.append(f"  canonical: {canonical}")

    for c in cols:
        if c == canonical:
            continue
        merge_report_lines.append(f"    merged: {c}")

        # Merge: canonical = max(canonical, c)
        df[canonical] = df[[canonical, c]].max(axis=1)

        # Drop synonym column from df
        df.drop(columns=[c], inplace=True)

    merge_report_lines.append("")

# Save report
with open(REPORT_TXT, "w", encoding="utf-8") as f:
    f.write("\n".join(merge_report_lines))

print(f"Merge report saved to: {REPORT_TXT}")

# -----------------------------
# 5. SAVE CLEANED DATA + SYMPTOM LIST
# -----------------------------
# Recompute symptom columns from df
final_symptom_cols = [c for c in df.columns if c not in NON_SYMPTOM_COLS]

# Put non-symptoms first for consistency
final_cols = NON_SYMPTOM_COLS + final_symptom_cols
df = df[final_cols]

df.to_csv(OUTPUT_CSV, index=False)
print(f"Merged dataset saved to: {OUTPUT_CSV}")
print(f"Final symptom columns: {len(final_symptom_cols)}")

with open(OUTPUT_SYMPTOM_LIST, "wb") as f:
    pickle.dump(final_symptom_cols, f)

print(f"Updated symptom list saved to: {OUTPUT_SYMPTOM_LIST}")
