"""
normalize_diseases.py

Finds near-duplicate disease names using sentence embeddings and renames
them to a single canonical name. Rows are KEPT — only the name is changed.
Only merges at very high similarity (threshold 0.97) to avoid false positives.

Input:  merged/merged_dataset.csv
Output: merged/merged_dataset.csv  (overwritten in place)
        merged/disease_name_report.txt
"""

import pickle
import pandas as pd
from pathlib import Path
from sentence_transformers import SentenceTransformer, util

MERGED_DIR  = Path("merged")
INPUT_CSV   = MERGED_DIR / "merged_dataset.csv"
REPORT_PATH = MERGED_DIR / "disease_name_report.txt"

SIM_THRESHOLD = 0.97  # very strict — only near-identical names

df = pd.read_csv(INPUT_CSV, low_memory=False)

disease_names = df["name"].unique().tolist()
print(f"Unique disease names before: {len(disease_names)}")

# embed disease names
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
embeddings = model.encode(disease_names, convert_to_tensor=True,
                           normalize_embeddings=True, show_progress_bar=True)

# union-find
sim_matrix = util.cos_sim(embeddings, embeddings)
n = len(disease_names)
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
            union(i, j)

clusters = {}
for i in range(n):
    clusters.setdefault(find(i), []).append(i)

groups = [idx for idx in clusters.values() if len(idx) > 1]
print(f"Found {len(groups)} synonym groups")

# build rename map: all names in a group → shortest name (canonical)
rename_map = {}
report_lines = []
for g_idx, indices in enumerate(groups, start=1):
    names = [disease_names[i] for i in indices]
    canonical = min(names, key=len)
    report_lines.append(f"Group {g_idx}: canonical='{canonical}'")
    for name in names:
        if name != canonical:
            rename_map[name] = canonical
            report_lines.append(f"    ← {name}")
    report_lines.append("")

# apply rename — rows are kept, only the name value changes
df["name"] = df["name"].replace(rename_map)

print(f"Renamed {len(rename_map)} disease name variants")
print(f"Unique disease names after:  {df['name'].nunique()}")

df.to_csv(INPUT_CSV, index=False)
print(f"Saved → {INPUT_CSV}")

with open(REPORT_PATH, "w", encoding="utf-8") as f:
    f.write("\n".join(report_lines))
print(f"Report → {REPORT_PATH}")
print("\nnormalize_diseases.py done ✓")