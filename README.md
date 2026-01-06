# Scalable Two-Stage Ranking Engine with Offline LLM Semantic Signals

## 🚀 Executive Summary
This repository implements a **production-grade recommendation pipeline** using a classic two-stage architecture: **Retrieval (Candidate Generation)** followed by **Re-ranking (Precision Scoring)**. 

The core innovation is the integration of **Offline LLM Semantic Signals**. By distilling Large Language Models into pre-computed semantic features, the system captures deep intent while maintaining a **sub-50ms inference latency**—making it suitable for high-traffic e-commerce and content platforms.

---

## 🏗 Architecture & System Design

The system follows the industry-standard "Funnel" approach to balance scale and precision:

### Stage 1: Retrieval (Candidate Generation)
*   **Algorithm:** Alternating Least Squares (ALS) Matrix Factorization.
*   **Target:** Scan $N$ items to find top-K candidates.
*   **Metric:** Optimized for **Recall**.
*   **Stack:** `implicit` library for high-performance vectorized operations.

### Stage 2: Re-ranking (Scoring)
*   **Algorithm:** Gradient Boosted Decision Trees (CatBoost).
*   **Features:** User history, item metadata, and **LLM Semantic Embeddings**.
*   **Metric:** Optimized for **NDCG** and **Precision@K**.
*   **LLM Integration:** Item descriptions were processed offline to extract semantic clusters and vector embeddings, allowing the model to understand "why" a user likes a category beyond simple ID matching.

---

## 💡 System Design Decisions: Why "Offline" LLM?

In Tier-1 production environments (Amazon, Shopify, Yandex), real-time LLM inference is often prohibited by:
1.  **Latency:** LLM p99 latency (500ms+) exceeds the budget for real-time ranking (<100ms).
2.  **Throughput:** Serving 10k+ requests/sec via LLMs is economically non-viable.

**The Solution:** This project treats the LLM as a **feature extractor** during the ETL/Offline phase. Embeddings are pre-computed and stored in an artifact layer (mocked in `models/artifacts/`). At runtime, the CatBoost model performs a simple feature lookup, delivering "LLM-level intelligence" at microsecond speeds.

---

## 📊 Evaluation Results

Evaluated on the official **MovieLens 100K (ua.test)** split (10 held-out interactions per user).

| Pipeline Stage | Recall@5 | NDCG@5 | Recall@10 | NDCG@10 | Recall@100 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Stage 1 (ALS Retrieval)** | 0.128 | 0.279 | 0.211 | 0.239 | 0.621 |
| **Stage 2 (CatBoost + LLM Signals)** | **0.136** | **0.291** | **0.217** | **0.246** | **0.621** |

**Key Takeaways:**
*   Adding the Re-ranking stage improved **NDCG@5 by ~4.3%**.
*   The LLM signals effectively "pushed" relevant items higher in the top-5 list, directly impacting potential conversion metrics.

---

## 📂 Project Structure

```text
two_stage_ranking_with_llm/
├── app.py                  # FastAPI service (Lifespan management)
├── models/
│   ├── stage1_candidate.py # Retrieval logic (ALS)
│   ├── stage2_rerank.py    # Ranking logic (CatBoost)
│   └── artifacts/          # Pre-computed features & serialized models
├── utils/
│   ├── stage2_feature_builders.py # Online feature assembly
│   └── llm_embedding.py           # Offline embedding generation logic
├── Dockerfile              # Containerization (Debian-slim)
├── deployment.yaml         # Kubernetes Manifest (Deployment/Service)
└── requirements.txt        # Managed dependencies
```

---

## 🛠 Setup & Deployment

### 1. Local Environment
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt 
python3 -m uvicorn app:app --host 0.0.0.0 --port 8000
```

### 2. Containerization (Docker)
```bash
docker build -t ranking-service .  
docker run -p 8000:8000 ranking-service
```

### 3. Orchestration (Kubernetes)
Designed for cloud-native environments (AWS EKS / GCP GKE).
```bash
kubectl apply -f deployment.yaml
kubectl get pods
```

---

## 📡 API Usage

**Request:** `GET /recommend?user_id=123&top_k=5`

**Response:**
```json
{
  "user_id": 123,
  "recommendations": [42, 17, 89, 3, 56],
  "status": "success", 
  "metadata": { "model_version": "2-stage-v1-llm" }
}
```

---

## 🚀 Future Roadmap
*   **Vector Database:** Integrate **Milvus/FAISS** for Stage 1 (ANN search).
*   **Feature Store:** Deploy **Redis/Feast** for sub-ms feature retrieval.
*   **Monitoring:** Add Prometheus metrics for **P95 Latency** and **Prediction Drift**.
*   **CI/CD:** Automated model re-training and deployment via GitHub Actions.

---
**Data Source:** [MovieLens 100K](https://grouplens.org/datasets/movielens/100k/). Using official `ua.base` (train) and `ua.test` (eval) splits.
