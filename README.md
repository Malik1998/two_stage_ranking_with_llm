# Two-Stage Ranking System with LLM Signals

## Overview

This project demonstrates a **production-inspired two-stage recommendation pipeline** with LLM-based signals and an API layer.

The goal is not to build a full production system, but to show **end-to-end ownership**:
candidate generation, re-ranking, evaluation, and model serving.

---

## Architecture

**Stage 1 — Candidate Generation**

* Generates a small set of relevant candidates per user
* Baseline approach (e.g. collaborative filtering / simple embeddings)
* Optimized for recall

**Stage 2 — Re-ranking**

* Feature-based re-ranking model
* Incorporates **LLM-generated signals** (e.g. semantic similarity between user profile and item descriptions)
* Optimized for ranking quality (NDCG / Recall@K)

**Serving Layer**

* FastAPI endpoint for online inference
* Single endpoint: `/recommend`

---

## Project Structure

```
two_stage_ranking_with_llm/
│
├── data/               # Sample dataset
├── models/
│   ├── stage1_candidate.py
│   └── stage2_rerank.py
├── utils/
│   ├── data_loader.py
│   └── llm_features.py
├── api/
│   └── app.py
│
├── requirements.txt
└── README.md
```

---

## API Example

**Endpoint**

```
GET /recommend?user_id=123&top_k=5
```

**Response**

```json
{
  "user_id": 123,
  "top_items": [42, 17, 89, 3, 56]
}
```

---

## Evaluation

Offline evaluation is used:

* Recall@K
* NDCG@K

Metrics are computed on a small validation split to demonstrate ranking quality improvements after re-ranking.

---

## LLM Signals

LLM-based features are used as **additional ranking signals**, such as:

* Semantic embeddings of item descriptions
* Similarity between user preferences and item content

LLM inference is simplified and partially mocked to focus on **system design rather than model scale**.

---

## Limitations (Intentional)

* Offline evaluation only
* Single-node execution
* No concurrency or caching
* No cold-start handling
* No online A/B testing

These limitations are intentional to keep the project focused and bounded.

---

## What Would Be Done in Production

* Pre-compute and cache embeddings
* Add feature store
* Introduce diversity & business rules
* Add latency monitoring and fallback logic
* Online A/B testing

---

## Status

 ⃣ Candidate generation implemented
 ⃣ Re-ranking with LLM signals implemented
 ⃣ FastAPI serving layer
 ⃣ Offline evaluation

**Project intentionally stopped at this point.**

---

## Why This Project

This repository is meant to demonstrate:

* Two-stage ranking architecture
* Practical LLM integration
* Ownership over the full ML lifecycle
* Ability to define scope and stop intentionally


## Setup
```
python3 -m venv create venv
source venv/bin/activate
pip3 install -r requirements.txt 

```

## Data
Вот **краткий, аккуратный и профессиональный вариант** для `README.md`.
Можно **копировать 1-в-1**.

---

## Data

The data was downloaded from the [MovieLens 100K dataset](https://grouplens.org/datasets/movielens/100k/).

For this project, only the official **`ua.base` / `ua.test` split** is used:

* `ua.base` is used for training
* `ua.test` is used for evaluation

This split contains **exactly 10 held-out interactions per user** in the test set and is provided by the dataset authors.
Using the official split helps avoid data leakage and provides a consistent evaluation setup for ranking metrics such as Recall@K and NDCG@K.

