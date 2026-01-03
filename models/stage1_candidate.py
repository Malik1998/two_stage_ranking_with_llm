import pickle
import numpy as np
from scipy.sparse import load_npz
from implicit.als import AlternatingLeastSquares

class CandidateGenerator:
    def __init__(self, artifacts_path="models/artifacts"):
        self.model = AlternatingLeastSquares()
        with open(f"{artifacts_path}/als_model.pkl", "rb") as f:
            self.model = pickle.load(f)

        self.matrix = load_npz(f"{artifacts_path}/interaction_matrix.npz")

        with open(f"{artifacts_path}/user_map.pkl", "rb") as f:
            self.user_map = pickle.load(f)

        with open(f"{artifacts_path}/item_map.pkl", "rb") as f:
            self.item_map = pickle.load(f)

        self.internal_to_item_id = {v: k for k, v in self.item_map.items()}

        self.user_id_to_internal = {k: v for k, v in self.user_map.items()}


    def recommend(self, user_id: int, top_n=100):
        if user_id not in self.user_id_to_internal:
            return []

        uid = self.user_id_to_internal[user_id]

        user_row = self.matrix[uid]

        items, scores = self.model.recommend(
            userid=uid,
            user_items=user_row,
            N=top_n,
            filter_already_liked_items=True
        )

        return [self.internal_to_item_id[i] for i in items]