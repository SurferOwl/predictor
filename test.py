import pandas as pd
df = pd.read_csv("merged/merged_dataset_synonyms.csv")
app = df[df["name"] == "diverticulosis"]
print(f"Appendicitis rows: {len(app)}")

# show which symptom columns are set to 1
symptom_cols = [c for c in df.columns if c not in ("code", "name")]
app_symptoms = app[symptom_cols].max()
active = app_symptoms[app_symptoms > 0].index.tolist()
print(f"\nActive symptom columns ({len(active)}):")
for s in active:
    print(f"  {s}")

ab_cols = [c for c in df.columns if "abdominal" in c]
print(ab_cols)