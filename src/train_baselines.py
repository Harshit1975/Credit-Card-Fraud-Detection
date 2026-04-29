import pandas as pd
import numpy as np
import os
import joblib
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score
from features import add_features
from pipeline import pre

def train_baselines():
    print("Training Baseline Models...")
    df = pd.read_parquet("data/transactions.parquet")
    cut = int(len(df)*0.8)
    tr, va = df.iloc[:cut], df.iloc[cut:]
    tr, va = add_features(tr), add_features(va)

    Xtr = tr.drop(columns=["is_fraud","ts","tx_id","merchant_id_hash","card_id_hash"])
    ytr = tr["is_fraud"]
    Xva = va.drop(columns=["is_fraud","ts","tx_id","merchant_id_hash","card_id_hash"])
    yva = va["is_fraud"]

    logit = Pipeline([("pre", pre), ("clf", LogisticRegression(max_iter=500, class_weight="balanced"))])
    rf    = Pipeline([("pre", pre), ("clf", RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=42))])

    for name, model in [("LogReg",logit),("RF",rf)]:
        model.fit(Xtr, ytr)
        p = model.predict_proba(Xva)[:,1]
        print(f"{name} PR-AUC: {average_precision_score(yva, p):.4f}")

if __name__ == "__main__":
    train_baselines()
