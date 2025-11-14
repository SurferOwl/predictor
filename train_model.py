# train.py
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
    classification_report,
)

# -----------------------------------------
# PATHS
# -----------------------------------------
DATA_PATH = Path("data/merged_dataset_synonyms.csv")
TEST_SAVE_PATH = Path("data/test_data.csv")
MODEL_SAVE_PATH = Path("data/disease_model.pkl")      # best model
SYMPTOM_LIST_PATH = Path("data/symptom_list.pkl")     # feature columns


# -----------------------------------------
# 1. LOAD SYMPTOM MATRIX
# -----------------------------------------
df = pd.read_csv(DATA_PATH)

# Features and target
X = df.drop(columns=["code", "name"], errors="ignore")
y = df["name"]

symptom_list = list(X.columns)  # keep feature order for later / predict.py

# -----------------------------------------
# 2. SPLIT UNIQUE vs DUPLICATED DISEASES
# -----------------------------------------
counts = y.value_counts()

duplicated_diseases = counts[counts >= 2].index.tolist()
unique_diseases = counts[counts == 1].index.tolist()

df_duplicated = df[df["name"].isin(duplicated_diseases)]
df_unique = df[df["name"].isin(unique_diseases)]

X_dup = df_duplicated.drop(columns=["code", "name"], errors="ignore")
y_dup = df_duplicated["name"]

X_unique = df_unique.drop(columns=["code", "name"], errors="ignore")
y_unique = df_unique["name"]

print(f"Duplicated diseases: {len(duplicated_diseases)}")
print(f"Unique diseases:     {len(unique_diseases)}")
print(f"Duplicated rows:     {len(df_duplicated)}")
print(f"Unique rows:         {len(df_unique)}")

# -----------------------------------------
# 3. TRAIN/TEST SPLIT ONLY FOR DUPLICATED DISEASES
# -----------------------------------------
X_train_dup, X_test, y_train_dup, y_test = train_test_split(
    X_dup, y_dup, test_size=0.2, random_state=42, stratify=y_dup
)

# Add unique diseases to training set only
X_train = pd.concat([X_train_dup, X_unique], ignore_index=True)
y_train = pd.concat([y_train_dup, y_unique], ignore_index=True)

print(f"Final train size: {len(X_train)}")
print(f"Final test size:  {len(X_test)}")

# -----------------------------------------
# 4. SAVE THE TEST SET (full rows)
# -----------------------------------------
test_df = df.loc[X_test.index].copy()
test_df.to_csv(TEST_SAVE_PATH, index=False)
print(f"Saved test data to: {TEST_SAVE_PATH}")


# -----------------------------------------
# 5. DEFINE MULTIPLE MODELS (extended)
# -----------------------------------------
def make_models():
    """
    Return dict: name -> estimator
    """
    models = {}

    # Trees / forests / boosting
    models["DecisionTree"] = DecisionTreeClassifier(
        random_state=42
    )

    models["RandomForest"] = RandomForestClassifier(
        n_estimators=300, random_state=42, n_jobs=-1
    )

    models["ExtraTrees"] = ExtraTreesClassifier(
        n_estimators=300, random_state=42, n_jobs=-1
    )

    models["GradientBoosting"] = GradientBoostingClassifier(random_state=42)

    models["AdaBoost"] = AdaBoostClassifier(
        n_estimators=300, random_state=42
    )

    # Linear / probabilistic models
    models["LogReg_ovr"] = LogisticRegression(
        max_iter=1000, n_jobs=-1, multi_class="ovr"
    )

    models["SGD_Log"] = SGDClassifier(
        loss="log_loss",  # logistic regression
        max_iter=1000,
        n_jobs=-1,
        random_state=42,
    )

    models["ComplementNB"] = ComplementNB()  # works well with 0/1 high-dim data

    # k-NN
    models["KNN_5"] = KNeighborsClassifier(
        n_neighbors=5,
        n_jobs=-1,
    )

    # Neural network
    models["MLP_NeuralNet"] = MLPClassifier(
        hidden_layer_sizes=(256, 128),
        activation="relu",
        max_iter=200,
        random_state=42,
    )

    # Soft voting ensemble
    voting_clf = VotingClassifier(
        estimators=[
            ("rf", RandomForestClassifier(n_estimators=200, random_state=42)),
            ("lr", LogisticRegression(max_iter=1000, n_jobs=-1, multi_class="ovr")),
            ("mlp", MLPClassifier(hidden_layer_sizes=(128,), max_iter=150, random_state=42)),
        ],
        voting="soft",
        n_jobs=-1,
    )
    models["Voting_Soft"] = voting_clf

    # Stacking ensemble
    stacking_clf = StackingClassifier(
        estimators=[
            ("rf", RandomForestClassifier(n_estimators=200, random_state=42)),
            ("lr", LogisticRegression(max_iter=1000, n_jobs=-1, multi_class="ovr")),
        ],
        final_estimator=LogisticRegression(max_iter=1000, n_jobs=-1),
        passthrough=True,
        n_jobs=-1,
    )
    models["Stacking"] = stacking_clf

    return models


# -----------------------------------------
# 6. EVALUATION FUNCTION
# -----------------------------------------
def evaluate_model(name, model, X_train, y_train, X_test, y_test):
    print(f"\n\n====================== {name} ======================")
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    f1_macro = f1_score(y_test, y_pred, average="macro", zero_division=0)
    f1_micro = f1_score(y_test, y_pred, average="micro", zero_division=0)
    f1_weighted = f1_score(y_test, y_pred, average="weighted", zero_division=0)
    ham = hamming_loss(y_test, y_pred)
    jacc_macro = jaccard_score(y_test, y_pred, average="macro", zero_division=0)
    jacc_weighted = jaccard_score(y_test, y_pred, average="weighted", zero_division=0)
    mcc = matthews_corrcoef(y_test, y_pred)

    print("Accuracy:        ", f"{acc:.4f}")
    print("F1 macro:        ", f"{f1_macro:.4f}")
    print("F1 micro:        ", f"{f1_micro:.4f}")
    print("F1 weighted:     ", f"{f1_weighted:.4f}")
    print("Hamming Loss:    ", f"{ham:.4f}")
    print("Jaccard (macro): ", f"{jacc_macro:.4f}")
    print("Jaccard (weight):", f"{jacc_weighted:.4f}")
    print("MCC:             ", f"{mcc:.4f}")

    # print("\nClassification report:")
    # print(classification_report(y_test, y_pred, zero_division=0))

    return {
        "name": name,
        "estimator": model,
        "accuracy": acc,
        "f1_macro": f1_macro,
        "f1_micro": f1_micro,
        "f1_weighted": f1_weighted,
        "hamming_loss": ham,
        "jaccard_macro": jacc_macro,
        "jaccard_weighted": jacc_weighted,
        "mcc": mcc,
    }


# -----------------------------------------
# 7. TRAIN & COMPARE ALL MODELS
# -----------------------------------------
models = make_models()
results = []

for name, est in models.items():
    metrics = evaluate_model(name, est, X_train, y_train, X_test, y_test)
    results.append(metrics)

# Show summary table
results_df = pd.DataFrame(results).drop(columns=["estimator"])
print("\n\n====================== SUMMARY TABLE ======================")
print(results_df.sort_values(by="f1_weighted", ascending=False))

# -----------------------------------------
# 8. PICK BEST MODEL & SAVE IT + SYMPTOM LIST
# -----------------------------------------
best_row = max(results, key=lambda r: r["f1_weighted"])
best_model = best_row["estimator"]
best_name = best_row["name"]

with open(MODEL_SAVE_PATH, "wb") as f:
    pickle.dump(best_model, f)

with open(SYMPTOM_LIST_PATH, "wb") as f:
    pickle.dump(symptom_list, f)

print(f"\nBest model: {best_name}")
print(f"Saved best model to:   {MODEL_SAVE_PATH}")
print(f"Saved symptom list to: {SYMPTOM_LIST_PATH}")
