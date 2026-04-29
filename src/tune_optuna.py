import optuna
import pandas as pd
import numpy as np
import joblib
import os
from sklearn.pipeline import Pipeline
from sklearn.metrics import average_precision_score, confusion_matrix
from xgboost import XGBClassifier
from features import add_features
from pipeline import pre

def tune_and_train():
    print("Loading data for Optuna tuning...")
    df = pd.read_parquet("data/transactions.parquet")
    cut = int(len(df)*0.8)
    tr, va = df.iloc[:cut], df.iloc[cut:]
    tr, va = add_features(tr), add_features(va)

    Xtr = tr.drop(columns=["is_fraud","ts","tx_id","merchant_id_hash","card_id_hash"])
    ytr = tr["is_fraud"]
    Xva = va.drop(columns=["is_fraud","ts","tx_id","merchant_id_hash","card_id_hash"])
    yva = va["is_fraud"]

    pos_weight = (len(ytr)-ytr.sum())/max(1,ytr.sum())

    def objective(trial):
        params = dict(
            n_estimators=trial.suggest_int("n_estimators", 100, 300),
            max_depth=trial.suggest_int("max_depth", 3, 6),
            learning_rate=trial.suggest_float("lr", 0.01, 0.2, log=True),
            subsample=trial.suggest_float("subsample", 0.6, 1.0),
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.6, 1.0),
            reg_lambda=trial.suggest_float("reg_lambda", 0, 5),
            reg_alpha=trial.suggest_float("reg_alpha", 0, 2),
            scale_pos_weight=pos_weight, 
            random_state=42, 
            n_jobs=-1
        )
        pipe = Pipeline([("pre", pre), ("xgb", XGBClassifier(**params))])
        pipe.fit(Xtr, ytr)
        p = pipe.predict_proba(Xva)[:,1]
        return average_precision_score(yva, p)

    # Fast trial run to make it feasible for a demo
    print("Starting optimization...")
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=5)
    best = study.best_params
    print("Best params:", best)

    print("Training final model with best params...")
    best["scale_pos_weight"] = pos_weight
    best["random_state"] = 42
    best["n_jobs"] = -1

    final = Pipeline([("pre", pre), ("xgb", XGBClassifier(**best))]).fit(Xtr, ytr)
    p = final.predict_proba(Xva)[:,1]

    def pick_threshold(p, y, fn_cost=5000, fp_cost=50):
        best_t, best_cost = 0.5, 1e18
        for t in np.linspace(0.01,0.99,99):
            yhat = (p>=t).astype(int)
            tn, fp, fn, tp = confusion_matrix(y, yhat).ravel()
            cost = fn*fn_cost + fp*fp_cost
            if cost < best_cost: best_cost, best_t = cost, t
        return best_t, best_cost
        
    t_star, cost_star = pick_threshold(p, yva)
    print("Chosen threshold:", round(t_star,3), "Cost:", int(cost_star))
    
    os.makedirs("models", exist_ok=True)
    joblib.dump({"model":final, "threshold":float(t_star)}, "models/fraud_xgb.joblib")
    print("Model saved to models/fraud_xgb.joblib")

if __name__ == "__main__":
    tune_and_train()
