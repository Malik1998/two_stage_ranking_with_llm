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
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# ----------------------
# Config
# ----------------------
ARTIFACTS = "models/artifacts"
TOP_N = 200
EPS = 1e-6
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
os.makedirs(ARTIFACTS, exist_ok=True)

# ----------------------
# Build embeddings if missing
# ----------------------
if not os.path.exists(f"{ARTIFACTS}/item_embeddings.npy"):
    build_item_embeddings(ARTIFACTS, OPENROUTER_API_KEY)

if not os.path.exists(f"{ARTIFACTS}/user_embeddings.npy"):
    build_user_embeddings(ARTIFACTS)

# ----------------------
# Load data
# ----------------------
cols = ["user_id", "item_id", "rating", "timestamp"]
train_als = pd.read_csv("data/train_als.csv", sep=",", names=cols, header=0)
future_labels = pd.read_csv("data/future_labels.csv", sep=",", names=cols, header=0)

# ----------------------
# Compute user mean from train_als
# ----------------------
train_user_mean = train_als.groupby("user_id")["rating"].mean().to_dict()

# ----------------------
# Prepare future labels dict
# ----------------------
future_ratings_dict = future_labels.groupby("user_id").apply(
    lambda g: dict(zip(g["item_id"], g["rating"]))
).to_dict()

# Compute item popularity (optional feature)
item_popularity = train_als.groupby("item_id").size().to_dict()
np.save(f"{ARTIFACTS}/item_popularity.npy", item_popularity)

# ----------------------
# Load Stage-1 ALS + embeddings
# ----------------------
stage1 = CandidateGenerator(artifacts_path=ARTIFACTS)

item_embeddings = np.load(f"{ARTIFACTS}/item_embeddings.npy", allow_pickle=True).item()
user_embeddings = np.load(f"{ARTIFACTS}/user_embeddings.npy", allow_pickle=True).item()

users = pd.read_csv("data/u.user", sep="|", names=["user_id", "age", "gender", "occupation", "zip"], header=None)
users = pd.get_dummies(users, columns=["gender", "occupation"])
users = users.set_index("user_id")
users = users.drop(columns=["zip"])

# ----------------------
# Build training data
# ----------------------
X_all = []
y_all = []
group = []

for user_id, future_items_dict in future_ratings_dict.items():
    valid_items = []
    als_scores = []
    targets = []

    if user_id not in train_user_mean:
        continue

    rating_mean = train_user_mean[user_id]

    for item_id, rating in future_items_dict.items():
        score = stage1.predict(user_id, item_id)
        if score is None:
            continue

        valid_items.append(item_id)
        als_scores.append(score)
        target = (rating / (rating_mean + EPS))
        targets.append(target)

    if not valid_items:
        continue

    # фичи
    X = build_features(
        user_id=user_id,
        candidate_items=valid_items,
        als_scores=als_scores,
        user_embeddings=user_embeddings,
        item_embeddings=item_embeddings,
        item_popularity=item_popularity,
        user_info=users
    )

    X_all.append(X)
    y_all.extend(targets)
    group.append(len(valid_items))

X_all = np.vstack(X_all)
y_all = np.array(y_all)

print("Train shape:", X_all.shape)
print("Positive rate / mean target:", y_all.mean())

# ----------------------
# Train LGBM Ranker
# ----------------------

X_train, X_val, y_train, y_val = train_test_split(
    X_all,
    y_all,
    test_size=0.2,
    random_state=42
)

ranker = lgb.LGBMRegressor(
    n_estimators=300,
    learning_rate=0.1,
    max_depth=5,
    min_child_samples=32,
    reg_alpha=0.1,
    reg_lambda=0.01,
    random_state=42
)

ranker.fit(
    X_train, y_train,
)

y_train_pred = ranker.predict(X_train)
y_val_pred = ranker.predict(X_val)

train_mse = mean_squared_error(y_train, y_train_pred)
val_mse = mean_squared_error(y_val, y_val_pred)

train_r2 = r2_score(y_train, y_train_pred)
val_r2 = r2_score(y_val, y_val_pred)

print(f"Train MSE: {train_mse:.4f}, R2: {train_r2:.4f}")
print(f"Val MSE: {val_mse:.4f}, R2: {val_r2:.4f}")

with open(f"{ARTIFACTS}/lgbm_ranker.pkl", "wb") as f:
    pickle.dump(ranker, f)