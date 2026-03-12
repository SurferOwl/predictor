# predict.py
import pickle
import pandas as pd
from pathlib import Path
from sentence_transformers import SentenceTransformer, util
import torch
import httpx

# ── paths ─────────────────────────────────────────────────────────────────────
MODEL_DIR = Path("model")

disease_model = pickle.load(open(MODEL_DIR / "disease_model.pkl", "rb"))
symptom_list  = pickle.load(open(MODEL_DIR / "symptom_list.pkl",  "rb"))

# ── embeddings (computed once at import) ──────────────────────────────────────
symptom_texts      = [s.replace("_", " ") for s in symptom_list]
embedder           = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
symptom_embeddings = embedder.encode(
    symptom_texts,
    convert_to_tensor=True,
    normalize_embeddings=True,
)


# ── 1. match one symptom entry to dataset columns ────────────────────────────

def match_symptoms(symptom_name, description="", top_k=5, score_threshold=0.4):
    query = symptom_name.strip()
    if description:
        query += ". " + description.strip()

    query_emb = embedder.encode(query, convert_to_tensor=True, normalize_embeddings=True)
    scores    = util.cos_sim(query_emb, symptom_embeddings)[0].cpu()
    top_k     = min(top_k, len(symptom_list))

    top_scores, top_indices = torch.topk(scores, k=top_k)

    return [
        {
            "column": symptom_list[int(idx)],
            "label":  symptom_texts[int(idx)],
            "score":  float(score.item()),
        }
        for score, idx in zip(top_scores, top_indices)
        if float(score.item()) >= score_threshold
    ]


# ── 2. severity → (top_k, threshold) ─────────────────────────────────────────

def severity_to_params(severity: int) -> tuple:
    """Matches frontend scale: 0=none, 1-3=mild, 4-6=moderate, 7-10=severe"""
    if severity <= 0:    return 0, 1.00   # don't activate anything
    elif severity <= 3:  return 1, 0.52   # mild   — best single match
    elif severity <= 6:  return 2, 0.45   # moderate — top 2
    else:                return 3, 0.38   # severe — top 3, looser threshold


# ── 3. main prediction function ───────────────────────────────────────────────

def prettify_name(name: str) -> str:
    return name.replace("_", " ").title()

def predict_disease_from_multiple_symptoms(
    symptom_entries,
    default_top_k_diseases=5,
):
    """
    Args:
        symptom_entries: list of dicts with keys:
            name        (str)
            description (str)
            severity    (int, 0-10)
    Returns:
        {
            "per_symptom_matches": [...],
            "disease_predictions": [{"disease": str, "probability": float}, ...]
        }
    """
    input_data          = {symptom: 0 for symptom in symptom_list}
    per_symptom_matches = []

    for entry in symptom_entries:
        name     = entry.get("name", "").strip()
        desc     = entry.get("description", "").strip()
        severity = int(entry.get("severity", 5))

        top_k, threshold = severity_to_params(severity)
        if top_k == 0:
            per_symptom_matches.append({"input": entry, "matches": []})
            continue

        matches = match_symptoms(name, desc, top_k=top_k, score_threshold=threshold)

        for m in matches:
            if m["column"] in input_data:
                input_data[m["column"]] = 1

        per_symptom_matches.append({"input": entry, "matches": matches})

    X     = pd.DataFrame([input_data])
    probs = disease_model.predict_proba(X)[0]
    classes = disease_model.classes_

    top_idx = probs.argsort()[::-1][:default_top_k_diseases]

    return {
        "per_symptom_matches": per_symptom_matches,
        "disease_predictions": [
            {"disease": prettify_name(classes[i]), "probability": round(float(probs[i]), 4)}
            for i in top_idx
        ],
    }


# ── 4. fetch symptoms from your backend ──────────────────────────────────────

async def fetch_user_symptoms(user_id: str):
    url = f"http://localhost:8080/api/symptom/getAllPastWeek/{user_id}"
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(url)
            response.raise_for_status()
            data = response.json()
        return [
            {
                "name":        item.get("name", ""),
                "description": item.get("description", ""),
                "severity":    int(item.get("severity", 5)),
            }
            for item in data
        ]
    except Exception as e:
        print(f"Error fetching symptoms: {e}")
        return []

# ── 5. quick test ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    user_symptoms = [
    {"name": "abdominal pain",            "description": "pain that begins near the navel and shifts to the lower right abdomen", "severity": 9},
    {"name": "loss of appetite",          "description": "sudden lack of desire to eat",                                           "severity": 7},
    {"name": "nausea",                    "description": "feeling sick to the stomach following the onset of abdominal pain",      "severity": 7},
    {"name": "vomiting",                  "description": "expelling stomach contents due to irritation of the digestive tract",    "severity": 6},
    {"name": "fever",                     "description": "mild to moderate increase in body temperature",                          "severity": 6},
    {"name": "abdominal tenderness",      "description": "pain when pressure is applied to the lower right abdomen",               "severity": 8},
    {"name": "constipation",              "description": "difficulty passing stool or infrequent bowel movements",                 "severity": 5},
    {"name": "bloating",                  "description": "swelling or fullness of the abdomen caused by gas buildup",              "severity": 5},
    {"name": "inability to pass gas",     "description": "difficulty releasing gas due to intestinal blockage or irritation",      "severity": 6},
    {"name": "pain when moving",          "description": "abdominal pain that worsens when walking, coughing, or moving",          "severity": 8}
]

    result = predict_disease_from_multiple_symptoms(
        symptom_entries=user_symptoms,
        default_top_k_diseases=5,
    )

    print("=== Matched symptom columns ===")
    for item in result["per_symptom_matches"]:
        inp = item["input"]
        print(f"\nUser symptom: {inp['name']} (severity={inp['severity']})")
        for m in item["matches"]:
            print(f"  -> {m['label']}  (col: {m['column']}, score: {m['score']:.3f})")

    print("\n=== Top disease predictions ===")
    for d in result["disease_predictions"]:
        print(f"  - {d['disease']}  ({d['probability']:.3f})")