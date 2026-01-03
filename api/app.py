from fastapi import FastAPI
from models.candidate_model import CandidateGenerator

app = FastAPI()

# 🔥 загружается при старте сервиса
candidate_generator = CandidateGenerator()

@app.get("/recommend")
def recommend(user_id: int, top_k: int = 10):
    items = candidate_generator.recommend(user_id, top_n=top_k)
    return {
        "user_id": user_id,
        "recommendations": items
    }
