from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

NUM = ["amount","log_amount","prev_24h_tx_count_card","prev_24h_amt_card",
       "prev_1h_tx_count_card","velocity_amt_1h","avg_tx_amt_24h","velocity_ratio",
       "hour","dayofweek"]
CAT = ["merchant_cat","city","country","device_type","channel","is_international",
       "is_night","is_weekend","merchant_cat_rare"]

num = Pipeline([("imp", SimpleImputer(strategy="median")),("sc", StandardScaler())])
cat = Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False))])

pre = ColumnTransformer([("num", num, NUM), ("cat", cat, CAT)])
