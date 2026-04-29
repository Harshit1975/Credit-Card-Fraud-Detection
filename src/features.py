import pandas as pd
import numpy as np

def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # ratios & logs
    df["log_amount"] = np.log1p(df["amount"])
    df["avg_tx_amt_24h"] = (df["prev_24h_amt_card"] / (df["prev_24h_tx_count_card"]+1e-3))
    df["velocity_ratio"] = df["velocity_amt_1h"] / (df["avg_tx_amt_24h"]+1e-3)
    df["is_weekend"] = df["dayofweek"].isin([5,6]).astype(int)
    
    # rare merchant category flag
    cat_counts = df["merchant_cat"].value_counts()
    df["merchant_cat_rare"] = df["merchant_cat"].map(lambda x: int(cat_counts.get(x,0)<50))
    return df
