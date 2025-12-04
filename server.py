from fastapi import FastAPI
from predict import predict_disease_from_multiple_symptoms, fetch_user_symptoms

app = FastAPI()

@app.get("/predictions/{user_id}")
async def get_predictions(user_id: str):
    # Await the async function!
    symptoms = await fetch_user_symptoms(user_id)
    
    # Now symptoms is a real list, not a coroutine
    result = predict_disease_from_multiple_symptoms(symptoms, default_top_k_diseases=5)
    return result
