import numpy as np


def recall_at_k(recommended_items, relevant_items, k):
    """
    recommended_items: list[int]
    relevant_items: set[int]
    """
    if not relevant_items:
        return 0.0

    recommended_k = recommended_items[:k]
    hits = len(set(recommended_k) & relevant_items)
    return hits / len(relevant_items)


def dcg_at_k(recommended_items, relevant_items, k):
    dcg = 0.0
    for idx, item in enumerate(recommended_items[:k]):
        if item in relevant_items:
            dcg += 1.0 / np.log2(idx + 2)
    return dcg


def ndcg_at_k(recommended_items, relevant_items, k):
    ideal_dcg = sum(
        1.0 / np.log2(i + 2)
        for i in range(min(len(relevant_items), k))
    )
    if ideal_dcg == 0:
        return 0.0

    return dcg_at_k(recommended_items, relevant_items, k) / ideal_dcg


def evaluate_user(recommended_items, relevant_items, k_values=(5, 10)):
    metrics = {}
    for k in k_values:
        metrics[f"recall@{k}"] = recall_at_k(
            recommended_items, relevant_items, k
        )
        metrics[f"ndcg@{k}"] = ndcg_at_k(
            recommended_items, relevant_items, k
        )
    return metrics
