import numpy as np
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

import os
import numpy as np
import pandas as pd

from utils.llm_embedding import LLMEmbedder


def build_item_embeddings(
    artifacts_path: str,
    api_key: str,
):
    print("Building item embeddings...")

    # --- load genres
    genres = pd.read_csv(
        "data/u.genre",
        sep="|",
        names=["genre", "id"],
        encoding="latin-1",
    )

    genre_map = dict(zip(genres.id, genres.genre))

    # --- load items
    items = pd.read_csv(
        "data/u.item",
        sep="|",
        encoding="latin-1",
        header=None,
    )

    genre_cols = list(range(5, 24))

    texts = []
    item_ids = []

    for _, row in items.iterrows():
        item_id = int(row[0])
        title = row[1]

        item_genres = [
            genre_map[i - 5]
            for i in genre_cols
            if row[i] == 1
        ]

        text = f"{title}. Genres: {', '.join(item_genres)}"

        item_ids.append(item_id)
        texts.append(text)

    embedder = LLMEmbedder(api_key=api_key)
    vectors = embedder.embed_texts(texts)

    item_embeddings = {
        iid: vec for iid, vec in zip(item_ids, vectors)
    }

    np.save(f"{artifacts_path}/item_embeddings.npy", item_embeddings)
    print(f"Saved {len(item_embeddings)} item embeddings")


def build_user_embeddings(
    artifacts_path: str,
    top_n: int = 5,
):
    print("Building user embeddings...")

    train_df = pd.read_csv(
        "data/train_als.csv",
    )

    item_embeddings = np.load(
        f"{artifacts_path}/item_embeddings.npy",
        allow_pickle=True,
    ).item()

    user_embeddings = {}
    
    default_emb = np.mean(
        list(item_embeddings.values()),
        axis=0
    )

    not_default_count = 0
    for user_id, g in train_df.groupby("user_id"):
        top_items = (
            g.sort_values("rating", ascending=False)
            .head(top_n)["item_id"]
        )
        embs = [
            item_embeddings[i]
            for i in top_items
            if i in item_embeddings
        ]

        if embs:
            user_embeddings[user_id] = np.mean(embs, axis=0)
            not_default_count += 1
        else:
            user_embeddings[user_id] = default_emb

    np.save(f"{artifacts_path}/user_embeddings.npy", user_embeddings)
    print(f"Saved {len(user_embeddings)} user embeddings, not_default_count={not_default_count}")



def cosine_sim(u, v):
    return float(cosine_similarity(u.reshape(1, -1), v.reshape(1, -1))[0][0])

def build_features(
    user_id,
    candidate_items,
    als_scores,
    user_embeddings,
    item_embeddings,
    item_popularity,
    user_info=None
):
    rows = []

    user_emb = user_embeddings.get(user_id)
    user_extra = user_info.loc[user_id].values if user_info is not None and user_id in user_info.index else []
    
    for item_id, als_score in zip(candidate_items, als_scores):
        item_emb = item_embeddings.get(item_id)

        llm_sim = (
            cosine_sim(user_emb, item_emb)
            if user_emb is not None and item_emb is not None
            else 0.0
        )

        rows.append([
            als_score,
            llm_sim,
            item_popularity.get(item_id, 0),
            *user_extra 
        ])

    return np.array(rows)
