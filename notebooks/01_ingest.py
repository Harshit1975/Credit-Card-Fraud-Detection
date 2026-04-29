import pandas as pd

def ingest_data():
    print("Ingesting data from CSV to Parquet...")
    SCHEMA = {
        "tx_id":"string",
        "ts":"string",
        "amount":"float64",
        "merchant_cat":"category",
        "merchant_id_hash":"string",
        "card_id_hash":"string",
        "city":"category",
        "country":"category",
        "device_type":"category",
        "channel":"category",
        "hour":"int64",
        "dayofweek":"int64",
        "prev_24h_tx_count_card":"float64",
        "prev_24h_amt_card":"float64",
        "prev_1h_tx_count_card":"float64",
        "velocity_amt_1h":"float64",
        "is_international":"bool",
        "is_night":"bool",
        "is_fraud":"int64"
    }

    try:
        df = pd.read_csv("data/transactions.csv")
        for col, dtype in SCHEMA.items():
            if dtype == 'category' or dtype == 'string':
                df[col] = df[col].astype(str)
            elif dtype == 'bool':
                df[col] = df[col].astype(bool)
            elif 'float' in dtype or 'int' in dtype:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        df["ts"] = pd.to_datetime(df["ts"])
        df = df.sort_values("ts").reset_index(drop=True)
        df.to_parquet("data/transactions.parquet")
        print(f"Data shape: {df.shape}, Fraud Rate: {df.is_fraud.mean():.4f}")
        print("Data successfully ingested and saved to Parquet.")
    except Exception as e:
        print(f"Error during ingestion: {e}")

if __name__ == "__main__":
    ingest_data()
