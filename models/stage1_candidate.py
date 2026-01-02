import pandas as pd
import numpy as np

class CandidateGenerator:
    def __init__(self, data_path="data/ua_base.csv"):
        self.data = pd.read_csv(data_path, colums=["user_id", "item_id", "rating", "timestamp"])
    
    def generate_candidates(self, user_id, top_n=5):
        # простой baseline: топ items по item_id
        user_data = self.data[self.data['user_id'] == user_id]
        top_items = user_data.sort_values('item_id').head(top_n)
        return top_items['item_id'].tolist()
