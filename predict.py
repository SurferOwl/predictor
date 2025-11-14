# predict.py
import pickle
import pandas as pd
from sentence_transformers import SentenceTransformer, util
import torch

# ----------------------------------------------------
# LOAD DISEASE MODEL + SYMPTOM LIST
# ----------------------------------------------------
disease_model = pickle.load(open("data/disease_model.pkl", "rb"))

# IMPORTANT: this must match what train.py saves
symptom_list = pickle.load(open("data/symptom_list.pkl", "rb"))

# Nice text versions of symptoms for embeddings
symptom_texts = [s.replace("_", " ") for s in symptom_list]

# Pretrained sentence embedding model
embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# Precompute embeddings for all symptom columns (once at import)
symptom_embeddings = embedder.encode(
    symptom_texts,
    convert_to_tensor=True,
    normalize_embeddings=True
)


# ----------------------------------------------------
# 1. MATCH ONE (name, description) TO SYMPTOM COLUMNS
# ----------------------------------------------------
def match_symptoms(symptom_name, description="", top_k=5, score_threshold=0.4):
    """
    For a single user symptom (name + description),
    return the best matching symptom columns.
    """
    # Build query
    query = symptom_name.strip()
    if description:
        query += ". " + description.strip()

    # Encode query
    query_emb = embedder.encode(
        query,
        convert_to_tensor=True,
        normalize_embeddings=True
    )

    # Cosine similarity with all symptom columns
    scores = util.cos_sim(query_emb, symptom_embeddings)[0]
    scores = scores.cpu()

    # Get top-k
    top_k = min(top_k, len(symptom_list))
    top_scores, top_indices = torch.topk(scores, k=top_k)

    results = []
    for score, idx in zip(top_scores, top_indices):
        score_val = float(score.item())
        if score_val < score_threshold:
            continue
        col_name = symptom_list[int(idx)]
        label = symptom_texts[int(idx)]
        results.append(
            {"column": col_name, "label": label, "score": score_val}
        )

    return results


# ----------------------------------------------------
# 2. MULTIPLE SYMPTOMS → ONE FEATURE ROW → DISEASES
# ----------------------------------------------------
def predict_disease_from_multiple_symptoms(symptom_entries,
                                           default_top_k_diseases=3):
    """
    symptom_entries: list of dicts like
      {
        "name": "fever",
        "description": "high temperature with chills",
        "severity": 3   # 1=mild, 2=moderate, 3=severe (optional)
      }
    """
    # Start with all-zero feature vector
    input_data = {symptom: 0 for symptom in symptom_list}

    per_symptom_matches = []

    for entry in symptom_entries:
        name = entry.get("name", "").strip()
        desc = entry.get("description", "").strip()
        severity = entry.get("severity", 2)  # default = moderate

        # Decide how many columns to match & how strict to be
        top_k_symptoms_each, score_threshold = severity_to_params(severity)

        matches = match_symptoms(
            symptom_name=name,
            description=desc,
            top_k=top_k_symptoms_each,
            score_threshold=score_threshold,
        )

        # Turn matched columns on (still 0/1 for the model)
        for m in matches:
            col = m["column"]
            if col in input_data:
                input_data[col] = 1

        per_symptom_matches.append(
            {
                "input": {
                    "name": name,
                    "description": desc,
                    "severity": severity,
                },
                "matches": matches,
            }
        )

    # Create one "row" for this patient
    X = pd.DataFrame([input_data])

    # Predict diseases
    probs = disease_model.predict_proba(X)[0]
    classes = disease_model.classes_

    sorted_idx = probs.argsort()[::-1]
    top_idx = sorted_idx[:default_top_k_diseases]

    disease_results = [
        {
            "disease": classes[i],
            "probability": float(probs[i])
        }
        for i in top_idx
    ]

    return {
        "per_symptom_matches": per_symptom_matches,
        "disease_predictions": disease_results,
    }


def severity_to_params(severity):
    """
    Map symptom severity (1–3) to:
      - top_k_symptoms_each
      - score_threshold
    """
    if severity <= 1:  # mild
        top_k = 1
        threshold = 0.50
    elif severity == 2:  # moderate
        top_k = 2
        threshold = 0.45
    else:  # severe (3 or more)
        top_k = 3
        threshold = 0.40
    return top_k, threshold


# ----------------------------------------------------
# 3. EXAMPLE USAGE
# ----------------------------------------------------
if __name__ == "__main__":
    user_symptoms = [
    {
        "name": "leg weakness",
        "description": "weakness starting in the feet and moving upwards",
        "severity": 3
    },
    {
        "name": "tingling",
        "description": "pins and needles feeling in both legs",
        "severity": 2
    },
    {
        "name": "difficulty walking",
        "description": "must use support to walk",
        "severity": 3
    },
    {
        "name": "reduced reflexes",
        "description": "kneecap and ankle reflexes are weaker than usual",
        "severity": 2
    },
    {
        "name": "back pain",
        "description": "dull back pain radiating down the legs",
        "severity": 2
    }
]

    result = predict_disease_from_multiple_symptoms(
        symptom_entries=user_symptoms,
        default_top_k_diseases=5
    )

    print("=== Matched symptom columns ===")
    for item in result["per_symptom_matches"]:
        print(f"\nUser symptom: {item['input']['name']} | {item['input']['description']}")
        for m in item["matches"]:
            print(f"  -> {m['label']}  (col: {m['column']}, score: {m['score']:.3f})")

    print("\n=== Top disease predictions ===")
    for d in result["disease_predictions"]:
        print(f"- {d['disease']} ({d['probability']:.3f})")
