from utils.eval import evaluate_user
import numpy as np
from models.stage1_candidate import CandidateGenerator
import pandas as pd
import numpy as np


cols = ["user_id", "item_id", "rating", "timestamp"]
test_df = pd.read_csv("data/ua.test", sep="\t", names=cols)

stage1_model = CandidateGenerator(artifacts_path="models/artifacts")

all_metrics = []

for user_id in test_df["user_id"].unique():
    relevant_items = set(test_df[test_df.user_id == user_id]["item_id"])

    candidates = stage1_model.recommend(user_id, top_n=100)
    # reranked = stage2_model.rerank(user_id, candidates)

    metrics = evaluate_user(
        recommended_items=candidates,
        relevant_items=relevant_items,
        k_values=(5, 10, 100)
    )
    all_metrics.append(metrics)

# aggregate
mean_metrics = {
    k: np.mean([m[k] for m in all_metrics])
    for k in all_metrics[0]
}
print(mean_metrics)
