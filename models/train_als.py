import pandas as pd
import pickle
import numpy as np
from scipy.sparse import csr_matrix, save_npz
from implicit.als import AlternatingLeastSquares
from train_utils import train_valid_split

eps = 1e-6
# Load data
cols = ["user_id", "item_id", "rating", "timestamp"]
train = pd.read_csv("data/ua.base", sep="\t", names=cols)
train_als, future_labels = train_valid_split(train, valid_ratio=0.1)

train_als.to_csv("data/train_als.csv", index=False)
future_labels.to_csv("data/future_labels.csv", index=False)

train_user_mean = train.groupby("user_id")["rating"].mean().reset_index()
train = train.merge(train_user_mean, on="user_id", suffixes=("", "_mean"))


train["interaction"] = ( train["rating"] / (train["rating_mean"] + eps)).clip(0.25, 4.0)


# Encode ids
user_codes = train.user_id.astype("category")
item_codes = train.item_id.astype("category")

user_map = dict(zip(user_codes.cat.categories, range(len(user_codes.cat.categories))))

item_map = dict(zip(item_codes.cat.categories, range(len(item_codes.cat.categories))))

user_ids = user_codes.cat.codes
item_ids = item_codes.cat.codes

# Sparse matrix
matrix = csr_matrix((train.interaction, (user_ids, item_ids)))

# Train ALS
model = AlternatingLeastSquares(
    factors=64,
    regularization=0.01,
    iterations=20
)
model.fit(matrix)

# Save artifacts
import pickle
with open("models/artifacts/als_model.pkl", "wb") as f:
    pickle.dump(model, f)
save_npz("models/artifacts/interaction_matrix.npz", matrix)

with open("models/artifacts/user_map.pkl", "wb") as f:
    pickle.dump(user_map, f)

with open("models/artifacts/item_map.pkl", "wb") as f:
    pickle.dump(item_map, f)

print("ALS model and artifacts saved.")
