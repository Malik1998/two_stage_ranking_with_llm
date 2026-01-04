import pandas as pd

def train_valid_split(df, valid_ratio=0.2):
    train_rows = []
    valid_rows = []

    for user_id, g in df.groupby("user_id"):
        g = g.sort_values("timestamp")  # temporal order
        split_idx = int(len(g) * (1 - valid_ratio))

        train_rows.append(g.iloc[:split_idx])
        valid_rows.append(g.iloc[split_idx:])

    train_df = pd.concat(train_rows)
    valid_df = pd.concat(valid_rows)
    return train_df, valid_df
