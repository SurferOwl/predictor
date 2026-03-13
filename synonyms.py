"""
synonyms.py

Uses sentence embeddings to find near-duplicate symptom columns and
merges them into one canonical column.

Input:  merged/merged_dataset.csv
Output: merged/merged_dataset_synonyms.csv
        merged/symptom_list.pkl
        merged/symptom_merge_report.txt
"""

import pickle
import pandas as pd
from pathlib import Path
from sentence_transformers import SentenceTransformer, util

MERGED_DIR = Path("merged")
INPUT_CSV  = MERGED_DIR / "merged_dataset.csv"
OUTPUT_CSV = MERGED_DIR / "merged_dataset_synonyms.csv"
SYMPTOM_LIST_PATH = MERGED_DIR / "symptom_list.pkl"
REPORT_PATH = MERGED_DIR / "symptom_merge_report.txt"

SIM_THRESHOLD = 0.92

NEVER_MERGE = {
    frozenset(["hunger", "loss_of_appetite"]),
    frozenset(["hunger", "decreased_appetite"]),
    frozenset(["hunger", "loss_of_appetite_lk"]),
    frozenset(["hunger", "loss_of_appetite_r"]),
    frozenset(["hunger", "decreased_appetite_lk"]),
    frozenset(["low_blood_pressure", "high_blood_pressure"]),
    frozenset(["low_blood_pressure_r", "high_blood_pressure"]),
    frozenset(["low_blood_pressure_r", "high_blood_pressure_r"]),
    frozenset(["fast_heart_rate", "slow_heart_rate"]),
    frozenset(["fast_heart_rate", "too_slow__bradycardia__heart_rate"]),
    frozenset(["rapid_heart_rate", "slow_heart_rate"]),
}

# ── load ──────────────────────────────────────────────────────────────────────

df = pd.read_csv(INPUT_CSV)
NON_SYMPTOM_COLS = [c for c in ["code", "name"] if c in df.columns]
symptom_cols = [c for c in df.columns if c not in NON_SYMPTOM_COLS]
orig_symptom_cols = symptom_cols.copy()

print(f"Loaded: {INPUT_CSV}")
print(f"  • {len(df)} rows")
print(f"  • {len(orig_symptom_cols)} symptom columns")

# ── embed ─────────────────────────────────────────────────────────────────────

symptom_texts = [c.replace("_", " ") for c in orig_symptom_cols]

print(f"\nEncoding {len(symptom_texts)} symptom names ...")
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
embeddings = model.encode(
    symptom_texts,
    convert_to_tensor=True,
    normalize_embeddings=True,
    show_progress_bar=True,
)

# ── find synonym groups (union-find) ──────────────────────────────────────────

print(f"\nComputing similarity matrix (threshold={SIM_THRESHOLD}) ...")
sim_matrix = util.cos_sim(embeddings, embeddings)
n = len(orig_symptom_cols)

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

for i in range(n):
    for j in range(i + 1, n):
        if float(sim_matrix[i, j].item()) >= SIM_THRESHOLD:
            pair = frozenset([orig_symptom_cols[i], orig_symptom_cols[j]])
            if pair not in NEVER_MERGE:
                union(i, j)

clusters = {}
for i in range(n):
    clusters.setdefault(find(i), []).append(i)

groups = [idx for idx in clusters.values() if len(idx) > 1]
print(f"  Found {len(groups)} synonym groups")

# ── merge groups into canonical columns ───────────────────────────────────────

report_lines = []
merged_count = 0

for g_idx, indices in enumerate(groups, start=1):
    cols = [orig_symptom_cols[i] for i in indices]
    cols = [c for c in cols if c in df.columns]
    if len(cols) <= 1:
        continue

    canonical = min(cols, key=len)   # shortest name wins
    report_lines.append(f"Group {g_idx}:  canonical='{canonical}'")

    for c in cols:
        if c == canonical:
            continue
        report_lines.append(f"    ← {c}")
        df[canonical] = df[[canonical, c]].max(axis=1)
        df.drop(columns=[c], inplace=True)
        merged_count += 1

    report_lines.append("")

print(f"  Merged {merged_count} columns into their canonicals")

# ── save ──────────────────────────────────────────────────────────────────────

final_symptom_cols = [c for c in df.columns if c not in NON_SYMPTOM_COLS]
final_cols = NON_SYMPTOM_COLS + final_symptom_cols
df = df[final_cols]

df.to_csv(OUTPUT_CSV, index=False)
print(f"\nSaved → {OUTPUT_CSV}  ({len(final_symptom_cols)} symptom columns remaining)")

with open(SYMPTOM_LIST_PATH, "wb") as f:
    pickle.dump(final_symptom_cols, f)
print(f"Saved → {SYMPTOM_LIST_PATH}")

with open(REPORT_PATH, "w", encoding="utf-8") as f:
    f.write("\n".join(report_lines))
print(f"Saved → {REPORT_PATH}")

print("\nstep3_synonyms.py done ✓")