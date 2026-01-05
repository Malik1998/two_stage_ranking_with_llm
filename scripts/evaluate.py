from utils.eval import evaluate_user
import numpy as np
from models.stage1_candidate import CandidateGenerator
from models.stage2_rerank import Stage2ReRanker
import pandas as pd
import numpy as np
from tqdm import tqdm


cols = ["user_id", "item_id", "rating", "timestamp"]
test_df = pd.read_csv("data/ua.test", sep="\t", names=cols)

stage1_model = CandidateGenerator(artifacts_path="models/artifacts")
stage2_model = Stage2ReRanker()

all_metrics_stage1 = []
all_metrics_stage2 = []

for user_id in tqdm(test_df["user_id"].unique(), total=len(test_df["user_id"].unique())):
    relevant_items = set(test_df[test_df.user_id == user_id]["item_id"])

    candidates, scores = stage1_model.recommend_with_scores(user_id, top_n=100)
    reranked = stage2_model.rerank(user_id, candidates, scores, top_k=100)

    all_metrics_stage1.append(evaluate_user(
        recommended_items=candidates,
        relevant_items=relevant_items,
        k_values=(5, 10, 100)
    ))
    all_metrics_stage2.append(evaluate_user(
        recommended_items=reranked,
        relevant_items=relevant_items,
        k_values=(5, 10, 100)
    ))

# aggregate
mean_metrics_stage1 = {
    k: np.mean([m[k] for m in all_metrics_stage1])
    for k in all_metrics_stage1[0]
}
print(f"mean_metrics_stage1: {mean_metrics_stage1}")


# aggregate
mean_metrics_stage2 = {
    k: np.mean([m[k] for m in all_metrics_stage2])
    for k in all_metrics_stage2[0]
}
print(f"mean_metrics_stage2: {mean_metrics_stage2}")