import numpy as np
import lightgbm as lgb
import pickle
from utils.stage2_feature_builders import build_features
import pandas as pd

class Stage2ReRanker:
    def __init__(self, artifacts_path="models/artifacts"):
        # Load LightGBM model
        with open(f"{artifacts_path}/lgbm_ranker.pkl", "rb") as f:
            self.model = pickle.load(f)

        # Load embeddings
        self.item_embeddings = np.load(
            f"{artifacts_path}/item_embeddings.npy",
            allow_pickle=True
        ).item()

        self.user_embeddings = np.load(
            f"{artifacts_path}/user_embeddings.npy",
            allow_pickle=True
        ).item()

        self.item_popularity = np.load(
            f"{artifacts_path}/item_popularity.npy",
            allow_pickle=True
        ).item()
        
        self.user_info = pd.read_csv("data/u.user", sep="|", names=["user_id", "age", "gender", "occupation", "zip"], header=None)
        self.user_info = pd.get_dummies(self.user_info, columns=["gender", "occupation"]).set_index("user_id")
        self.user_info = self.user_info.drop(columns=["zip"])
    def rerank(
        self,
        user_id: int,
        candidate_items: list[int],
        als_scores: list[float],
        top_k: int = 10
    ) -> list[int]:
        """
        Re-rank candidates using LGBM LambdaRank model
        """

        if user_id not in self.user_embeddings:
            # fallback: keep ALS order
            return candidate_items[:top_k]

        X = build_features(
            user_id=user_id,
            candidate_items=candidate_items,
            als_scores=als_scores,
            user_embeddings=self.user_embeddings,
            item_embeddings=self.item_embeddings,
            item_popularity=self.item_popularity,
            user_info=self.user_info

        )

        scores = self.model.predict(X)

        ranked = sorted(
            zip(candidate_items, scores),
            key=lambda x: x[1],
            reverse=True
        )

        return [item for item, _ in ranked[:top_k]]
