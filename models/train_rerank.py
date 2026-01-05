import os
import numpy as np
import pandas as pd
import lightgbm as lgb
import pickle

from models.stage1_candidate import CandidateGenerator
from utils.stage2_feature_builders import (
    build_item_embeddings,
    build_user_embeddings,
    build_features
)
import random
from catboost import CatBoostRanker , Pool

# ----------------------
# Config
# ----------------------
ARTIFACTS = "models/artifacts"
TOP_N = 125
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

TOP_N_USER_EMBEDDINGS = 5
os.makedirs(ARTIFACTS, exist_ok=True)

# ----------------------
# Build embeddings if missing
# ----------------------
if not os.path.exists(f"{ARTIFACTS}/item_embeddings.npy"):
    build_item_embeddings(ARTIFACTS, OPENROUTER_API_KEY)

if not os.path.exists(f"{ARTIFACTS}/user_embeddings_top{TOP_N_USER_EMBEDDINGS}.npy"):
    build_user_embeddings(ARTIFACTS, top_n=TOP_N_USER_EMBEDDINGS)

# ----------------------
# Load data
# ----------------------
cols = ["user_id", "item_id", "rating", "timestamp"]
train_als = pd.read_csv("data/train_als.csv", sep=",", names=cols, header=0)
future_labels = pd.read_csv("data/future_labels.csv", sep=",", names=cols, header=0)


# ----------------------
# Prepare future labels dict
# ----------------------
future_ratings_dict = future_labels.groupby("user_id")[["item_id", "rating"]].apply(
    lambda g: dict(zip(g.item_id, g.rating))
).to_dict()

# Compute item popularity (optional feature)
item_popularity = train_als.groupby("item_id").size().to_dict()
np.save(f"{ARTIFACTS}/item_popularity.npy", item_popularity)

# ----------------------
# Load Stage-1 ALS + embeddings
# ----------------------
stage1 = CandidateGenerator(artifacts_path=ARTIFACTS)

item_embeddings = np.load(f"{ARTIFACTS}/item_embeddings.npy", allow_pickle=True).item()
user_embeddings = np.load(f"{ARTIFACTS}/user_embeddings_top{TOP_N_USER_EMBEDDINGS}.npy", allow_pickle=True).item()

users = pd.read_csv("data/u.user", sep="|", names=["user_id", "age", "gender", "occupation", "zip"], header=None)
users = pd.get_dummies(users, columns=["gender", "occupation"])
users = users.set_index("user_id")
users = users.drop(columns=["zip"])

items = pd.read_csv("data/u.item", sep="|", names=["item_id", "movie title" , "release date" , "video release date" ,
              "IMDb URL" , "unknown" , "Action" , "Adventure" , "Animation" ,
              "Children's" , "Comedy", "Crime" , "Documentary" , "Drama" , "Fantasy" ,
              "Film-Noir" , "Horror" , "Musical" , "Mystery" , "Romance" , "Sci-Fi" ,
              "Thriller" , "War" , "Western"], header=None, encoding="ISO-8859-1")
items = items.set_index("item_id")
items = items.drop(columns=["movie title", "release date" , "video release date", "IMDb URL" ])

# ----------------------
# Build training data
# ----------------------
X_train, X_val = [], []
y_train, y_val = [], []
group_train, group_val = [], []

item_popularity_sorted = sorted(item_popularity.items(), key=lambda x: x[1], reverse=True)
for user_id, future_items_dict in future_ratings_dict.items():
    valid_items = []
    als_scores = []
    targets = []
    
    stage1_items, stage1_scores = stage1.recommend_with_scores(user_id, top_n=TOP_N)
    for item_id, score in zip(stage1_items, stage1_scores):
        target = future_items_dict.get(item_id, 0) >= 4  # binary target: relevant if rating >=4
        targets.append(target)
        valid_items.append(item_id)
        als_scores.append(score)

    if not valid_items:
        continue

    X = build_features(
        user_id=user_id,
        candidate_items=valid_items,
        als_scores=als_scores,
        user_embeddings=user_embeddings,
        item_embeddings=item_embeddings,
        item_popularity=item_popularity,
        user_info=users,
        item_info=items
    )
    
    if random.random() < 0.8:
        X_train.append(X)
        y_train.extend(targets)
        group_train.extend([user_id] * len(valid_items))
    else:
        X_val.append(X)
        y_val.extend(targets)
        group_val.extend([user_id] * len(valid_items))


X_train, X_val = np.vstack(X_train), np.vstack(X_val)
y_train, y_val = np.array(y_train), np.array(y_val)

print("Train shape:", X_train.shape, " | Train mean target:", y_train.mean())

print("Val shape:", X_val.shape, " | Val mean target:", y_val.mean())


train_pool = Pool(data=X_train, label=y_train, group_id=group_train)
val_pool = Pool(data=X_val, label=y_val, group_id=group_val)

model = CatBoostRanker(
    iterations=1000,
    learning_rate=0.05,
    depth=5,
    l2_leaf_reg=3.0,
    loss_function='YetiRank',
    one_hot_max_size=10,
    verbose=50,
    min_child_samples=32,
    eval_metric='NDCG:top=10;hints=skip_train~false'
)

model.fit(
    train_pool,
    eval_set=val_pool,
    early_stopping_rounds=200
)

model.save_model(f"{ARTIFACTS}/catboost_ranker.cbm")