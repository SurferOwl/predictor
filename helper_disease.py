import pandas as pd

# --- READ INPUT CSV ---
df = pd.read_csv("data/diseases.csv")

# --- CLEAN & SPLIT SYMPTOMS INTO LISTS ---
df['Symptoms_list'] = df['Symptoms'].apply(
    lambda x: [s.strip() for s in str(x).split(",")]
)

# --- COLLECT ALL UNIQUE SYMPTOMS ---
all_symptoms = sorted({
    symptom
    for symptoms in df['Symptoms_list']
    for symptom in symptoms
})

# --- CREATE 0/1 COLUMNS FOR EACH SYMPTOM ---
for symptom in all_symptoms:
    df[symptom] = df['Symptoms_list'].apply(
        lambda lst: 1 if symptom in lst else 0
    )

# --- FINAL RESULT (Code, Name, Symptoms as binary columns) ---
final = df[['Code', 'Name'] + all_symptoms]

# --- SAVE OUTPUT ---
final.to_csv("data/disease_symptom_matrix.csv", index=False)

print("Saved to data/disease_symptom_matrix.csv")
