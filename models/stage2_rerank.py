import numpy as np
from utils.stage2_feature_builders import build_features
import pandas as pd
from catboost import CatBoostRanker


class Stage2ReRanker:
    def __init__(self, artifacts_path="models/artifacts", top_n_user_embeddings=5):
        self.model = CatBoostRanker()
        self.model.load_model(f"{artifacts_path}/catboost_ranker.cbm")
        # Load embeddings
        self.item_embeddings = np.load(
            f"{artifacts_path}/item_embeddings.npy",
            allow_pickle=True
        ).item()

        self.user_embeddings = np.load(
            f"{artifacts_path}/user_embeddings_top{top_n_user_embeddings}.npy",
            allow_pickle=True
        ).item()

        self.item_popularity = np.load(
            f"{artifacts_path}/item_popularity.npy",
            allow_pickle=True
        ).item()
        
        self.user_info = pd.read_csv("data/u.user", sep="|", names=["user_id", "age", "gender", "occupation", "zip"], header=None)
        self.user_info = pd.get_dummies(self.user_info, columns=["gender", "occupation"]).set_index("user_id")
        self.user_info = self.user_info.drop(columns=["zip"])
        
        self.items = pd.read_csv("data/u.item", sep="|", names=["item_id", "movie title" , "release date" , "video release date" ,
              "IMDb URL" , "unknown" , "Action" , "Adventure" , "Animation" ,
              "Children's" , "Comedy", "Crime" , "Documentary" , "Drama" , "Fantasy" ,
              "Film-Noir" , "Horror" , "Musical" , "Mystery" , "Romance" , "Sci-Fi" ,
              "Thriller" , "War" , "Western"], header=None, encoding="ISO-8859-1")
        self.items = self.items.set_index("item_id")
        self.items = self.items.drop(columns=["movie title", "release date" , "video release date", "IMDb URL"])
        

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
            user_info=self.user_info,
            item_info=self.items

        )

        scores = self.model.predict(X)
        if user_id % 1000 == 0:
            print(f"Rerank scores: {scores}")

        ranked = sorted(
            zip(candidate_items, scores),
            key=lambda x: x[1],
            reverse=True
        )

        return [item for item, _ in ranked[:top_k]]
