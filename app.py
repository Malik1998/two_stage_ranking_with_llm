import logging
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
from contextlib import asynccontextmanager


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

models = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Loading ML models...")
    try:
        from models.stage1_candidate import CandidateGenerator
        from models.stage2_rerank import Stage2ReRanker
        
        models["candidate_gen"] = CandidateGenerator()
        models["reranker"] = Stage2ReRanker()
        logger.info("Models loaded successfully.")
    except Exception as e:
        logger.error(f"Failed to load models: {e}")
        raise e
    yield
    models.clear()

app = FastAPI(lifespan=lifespan)


class RecommendationResponse(BaseModel):
    user_id: int
    recommendations: List[int]
    status: str = "success"

@app.get("/", tags=["Health"])
def health_check():
    return {"status": "Ranking Service is Online", "models_loaded": len(models) > 0}


@app.get("/recommend", response_model=RecommendationResponse)
async def recommend(user_id: int, top_k: int = 10):
    if user_id < 0:
        raise HTTPException(status_code=400, detail="Invalid User ID")

    try:
        # 2. Stage 1: Candidate Generation (Retrieval)
        logger.info(f"Generating candidates for user {user_id}")
        candidates, scores = models["candidate_gen"].recommend_with_scores(user_id, top_n=top_k * 5)
        
        if not candidates:
            return RecommendationResponse(user_id=user_id, recommendations=[], status="no_candidates")

        # 3. Stage 2: Re-ranking
        logger.info(f"Reranking {len(candidates)} items for user {user_id}")
        final_items = models["reranker"].rerank(user_id, candidates, top_k=top_k, als_scores=scores)

        return RecommendationResponse(
            user_id=user_id,
            recommendations=final_items
        )

    except Exception as e:
        logger.error(f"Error during recommendation: {e}")
        raise HTTPException(status_code=500, detail="Internal Ranking Error")