import pandas as pd

cols = ["user_id", "item_id", "rating", "timestamp"]

train = pd.read_csv(
    "data/ua.base",
    sep="\t",
    names=cols
)

test = pd.read_csv(
    "data/ua.test",
    sep="\t",
    names=cols
)

print(train.head())
print(test.head())

print("Train interactions:", len(train))
print("Test interactions:", len(test))
print("Users in train:", train.user_id.nunique())
print("Users in test:", test.user_id.nunique())

print("--------------------------------")
# sanity check
train_per_user = train.groupby("user_id").size().value_counts()
print("train_per_user", train_per_user)
print("--------------------------------")
test_per_user = test.groupby("user_id").size().value_counts()
print("test_per_user", test_per_user)
