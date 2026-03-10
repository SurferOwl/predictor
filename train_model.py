"""
train_model.py

Trains multiple classifiers on the merged synonym-reduced dataset,
picks the best by weighted F1, and saves everything to model/.

Input:  merged/merged_dataset_synonyms.csv
        merged/symptom_list.pkl
Output: model/disease_model.pkl
        model/symptom_list.pkl     (copy — so predict.py has one source of truth)
        model/test_data.csv
        model/label_classes.pkl
"""

import warnings
warnings.filterwarnings("ignore")

import pickle
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    VotingClassifier,
    StackingClassifier,
    AdaBoostClassifier,
)
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import ComplementNB
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    hamming_loss,
    jaccard_score,
    matthews_corrcoef,
)

MERGED_DIR   = Path("merged")
MODEL_DIR    = Path("model")
MODEL_DIR.mkdir(exist_ok=True)

INPUT_CSV         = MERGED_DIR / "merged_dataset_synonyms.csv"
SYMPTOM_LIST_SRC  = MERGED_DIR / "symptom_list.pkl"
MODEL_SAVE_PATH   = MODEL_DIR  / "disease_model.pkl"
SYMPTOM_LIST_DST  = MODEL_DIR  / "symptom_list.pkl"
TEST_DATA_PATH    = MODEL_DIR  / "test_data.csv"
LABEL_CLASSES_PATH = MODEL_DIR / "label_classes.pkl"

# ── 1. load ───────────────────────────────────────────────────────────────────

df = pd.read_csv(INPUT_CSV)
X = df.drop(columns=["code", "name"], errors="ignore")
y = df["name"]
symptom_list = list(X.columns)

print(f"Dataset: {df.shape}  |  diseases: {y.nunique()}  |  symptoms: {len(symptom_list)}")

# ── 2. split (handle unique-only diseases) ───────────────────────────────────

counts = y.value_counts()
duplicated = counts[counts >= 2].index.tolist()
unique     = counts[counts == 1].index.tolist()

df_dup = df[df["name"].isin(duplicated)]
df_uni = df[df["name"].isin(unique)]

X_dup = df_dup.drop(columns=["code", "name"], errors="ignore")
y_dup = df_dup["name"]
X_uni = df_uni.drop(columns=["code", "name"], errors="ignore")
y_uni = df_uni["name"]

print(f"Duplicated diseases: {len(duplicated)}  ({len(df_dup)} rows)")
print(f"Unique diseases:     {len(unique)}       ({len(df_uni)} rows)")

if len(duplicated) >= 2:
    # ── preferred path: stratified split on duplicated rows only ──
    X_train_dup, X_test, y_train_dup, y_test = train_test_split(
        X_dup, y_dup, test_size=0.2, random_state=42, stratify=y_dup
    )
    X_train = pd.concat([X_train_dup, X_uni], ignore_index=True)
    y_train = pd.concat([y_train_dup, y_uni], ignore_index=True)
    # test_df lives in df_dup index space
    test_df = df.loc[X_test.index].copy()
else:
    # ── fallback: all diseases are unique — plain random split ──
    print("  ⚠ No duplicated diseases found; using plain random split on all data.")
    X_all = df.drop(columns=["code", "name"], errors="ignore")
    y_all = df["name"]
    X_train, X_test, y_train, y_test = train_test_split(
        X_all, y_all, test_size=0.2, random_state=42
        # no stratify — each class has only 1 sample
    )
    test_df = df.loc[X_test.index].copy()

print(f"Train: {len(X_train)}  |  Test: {len(X_test)}")

test_df.to_csv(TEST_DATA_PATH, index=False)
print(f"Test data saved → {TEST_DATA_PATH}")

# ── 3. model definitions ─────────────────────────────────────────────────────

TRAIN_SAMPLE = 50_000
if len(X_train) > TRAIN_SAMPLE:
    X_train, _, y_train, _ = train_test_split(
        X_train, y_train, train_size=TRAIN_SAMPLE, random_state=42
        # no stratify here — some classes have only 1 sample
    )
    print(f"Subsampled train to {TRAIN_SAMPLE} rows")

def make_models():
    return {
        "DecisionTree": DecisionTreeClassifier(random_state=42),

        "RandomForest": RandomForestClassifier(
            n_estimators=100, random_state=42, n_jobs=-1
        ),
        "ExtraTrees": ExtraTreesClassifier(
            n_estimators=100, random_state=42, n_jobs=-1
        ),
        #"LogReg_ovr": LogisticRegression(
         #   max_iter=300, n_jobs=-1, multi_class="ovr"
        #),
        "SGD_Log": SGDClassifier(
            loss="log_loss", max_iter=100, n_jobs=-1, random_state=42
        ),
        "ComplementNB": ComplementNB(),
        #"MLP_NeuralNet": MLPClassifier(
         #   hidden_layer_sizes=(128, ), activation="relu",
          #  max_iter=50, random_state=42
        #)
    }

# ── 4. train & evaluate ───────────────────────────────────────────────────────

def evaluate(name, model):
    print(f"\n{'='*20} {name} {'='*20}")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    metrics = {
        "name":          name,
        "estimator":     model,
        "accuracy":      accuracy_score(y_test, y_pred),
        "f1_macro":      f1_score(y_test, y_pred, average="macro",    zero_division=0),
        "f1_micro":      f1_score(y_test, y_pred, average="micro",    zero_division=0),
        "f1_weighted":   f1_score(y_test, y_pred, average="weighted", zero_division=0),
        "hamming_loss":  hamming_loss(y_test, y_pred),
        "jaccard_macro": jaccard_score(y_test, y_pred, average="macro",    zero_division=0),
        "jaccard_weighted": jaccard_score(y_test, y_pred, average="weighted", zero_division=0),
        "mcc":           matthews_corrcoef(y_test, y_pred),
    }

    for k, v in metrics.items():
        if k not in ("name", "estimator"):
            print(f"  {k:<22}: {v:.4f}")

    return metrics

results = [evaluate(n, m) for n, m in make_models().items()]

# ── 5. summary & save best ────────────────────────────────────────────────────

summary = pd.DataFrame(results).drop(columns=["estimator"])
print("\n\n" + "="*55)
print("SUMMARY TABLE (sorted by f1_weighted)")
print("="*55)
print(summary.sort_values("f1_weighted", ascending=False).to_string(index=False))

best = max(results, key=lambda r: r["f1_weighted"])
print(f"\nBest model: {best['name']}  (f1_weighted={best['f1_weighted']:.4f})")

# save model
with open(MODEL_SAVE_PATH, "wb") as f:
    pickle.dump(best["estimator"], f)
print(f"Saved → {MODEL_SAVE_PATH}")

# copy symptom list into model/ so predict.py only needs to look in one place
with open(SYMPTOM_LIST_SRC, "rb") as f:
    symptom_list_data = pickle.load(f)
with open(SYMPTOM_LIST_DST, "wb") as f:
    pickle.dump(symptom_list_data, f)
print(f"Saved → {SYMPTOM_LIST_DST}")

# save label classes
with open(LABEL_CLASSES_PATH, "wb") as f:
    pickle.dump(list(best["estimator"].classes_), f)
print(f"Saved → {LABEL_CLASSES_PATH}")

print("\nstep4_train.py done ✓")