import warnings
warnings.filterwarnings("ignore")
import pickle
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    hamming_loss,
    jaccard_score,
    matthews_corrcoef,
    top_k_accuracy_score
)

# ------------------------------------
# Load TEST data and model
# ------------------------------------
test_df = pd.read_csv("data/test_data.csv")
symptom_list = pickle.load(open("data/symptom_list.pkl", "rb"))

X_test = test_df[symptom_list]
y_test = test_df["name"]

with open("data/disease_model.pkl", "rb") as f:
    model = pickle.load(f)

print(f"Loaded test set: X_test={X_test.shape}, y_test={y_test.shape}")

# ------------------------------------
# Predictions on TEST set only
# ------------------------------------
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)

# ------------------------------------
# Metrics
# ------------------------------------
print("========== MODEL PERFORMANCE ON TEST SET ==========\n")

print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")

print("\n---- Precision ----")
print(f"Macro:    {precision_score(y_test, y_pred, average='macro', zero_division=0):.4f}")
print(f"Micro:    {precision_score(y_test, y_pred, average='micro', zero_division=0):.4f}")
print(f"Weighted: {precision_score(y_test, y_pred, average='weighted', zero_division=0):.4f}")

print("\n---- Recall ----")
print(f"Macro:    {recall_score(y_test, y_pred, average='macro', zero_division=0):.4f}")
print(f"Micro:    {recall_score(y_test, y_pred, average='micro', zero_division=0):.4f}")
print(f"Weighted: {recall_score(y_test, y_pred, average='weighted', zero_division=0):.4f}")

print("\n---- F1-Score ----")
print(f"Macro:    {f1_score(y_test, y_pred, average='macro', zero_division=0):.4f}")
print(f"Micro:    {f1_score(y_test, y_pred, average='micro', zero_division=0):.4f}")
print(f"Weighted: {f1_score(y_test, y_pred, average='weighted', zero_division=0):.4f}")

print("\n---- Other Metrics ----")
print(f"Hamming Loss:             {hamming_loss(y_test, y_pred):.4f}")
print(f"Jaccard Score (macro):    {jaccard_score(y_test, y_pred, average='macro', zero_division=0):.4f}")
print(f"Jaccard Score (weighted): {jaccard_score(y_test, y_pred, average='weighted', zero_division=0):.4f}")
print(f"Matthews Correlation Coefficient (MCC): {matthews_corrcoef(y_test, y_pred):.4f}")

# ------------------------------------
# Top-3 accuracy (pass full class list!)
# ------------------------------------
classes = model.classes_
top3_acc = top_k_accuracy_score(
    y_test,
    y_proba,
    k=3,
    labels=classes  # <-- key fix here
)
print(f"\nTop-3 Accuracy: {top3_acc:.4f}")

print("\n====================================================")
