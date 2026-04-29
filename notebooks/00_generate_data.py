import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import uuid
import os

def generate_synthetic_transactions(num_records=50000, fraud_rate=0.015):
    print("Generating synthetic transactions...")
    np.random.seed(42)
    
    # Generate dates over a 30 day period
    start_date = datetime(2023, 1, 1)
    timestamps = [start_date + timedelta(minutes=int(np.random.normal(i*1.5, 60))) for i in range(num_records)]
    timestamps.sort()
    
    # 500 unique cards and 100 unique merchants
    cards = [uuid.uuid4().hex[:10] for _ in range(500)]
    merchants = [uuid.uuid4().hex[:10] for _ in range(100)]
    
    data = []
    
    for i in range(num_records):
        is_fraud = np.random.rand() < fraud_rate
        
        card = np.random.choice(cards)
        merchant = np.random.choice(merchants)
        ts = timestamps[i]
        
        if not is_fraud:
            # Normal behavior
            amount = np.random.lognormal(mean=3.0, sigma=1.0) # $20 avg
            merchant_cat = np.random.choice(['grocery', 'gas', 'dining', 'retail'])
            city = np.random.choice(['New York', 'Chicago', 'San Francisco', 'Austin'])
            country = 'US'
            device_type = np.random.choice(['mobile', 'desktop'])
            channel = np.random.choice(['in_store', 'online'])
            is_international = False
            is_night = ts.hour < 6 or ts.hour > 22
        else:
            # Fraud behavior
            amount = np.random.lognormal(mean=6.0, sigma=1.5) # $400 avg
            merchant_cat = np.random.choice(['electronics', 'jewelry', 'travel', 'retail'])
            city = np.random.choice(['London', 'Paris', 'New York', 'Dubai'])
            country = np.random.choice(['UK', 'FR', 'US', 'AE'])
            device_type = np.random.choice(['mobile', 'unknown'])
            channel = 'online'
            is_international = country != 'US'
            is_night = np.random.rand() < 0.7 # 70% chance fraud happens at night
            
        data.append({
            'tx_id': uuid.uuid4().hex,
            'ts': ts.strftime('%Y-%m-%d %H:%M:%S'),
            'amount': round(amount, 2),
            'merchant_cat': merchant_cat,
            'merchant_id_hash': merchant,
            'card_id_hash': card,
            'city': city,
            'country': country,
            'device_type': device_type,
            'channel': channel,
            'hour': ts.hour,
            'dayofweek': ts.weekday(),
            'prev_24h_tx_count_card': np.random.randint(0, 5) if not is_fraud else np.random.randint(3, 15),
            'prev_24h_amt_card': round(np.random.uniform(10, 200)) if not is_fraud else round(np.random.uniform(500, 5000)),
            'prev_1h_tx_count_card': np.random.randint(0, 2) if not is_fraud else np.random.randint(1, 5),
            'velocity_amt_1h': round(np.random.uniform(0, 50)) if not is_fraud else round(np.random.uniform(200, 2000)),
            'is_international': is_international,
            'is_night': is_night,
            'is_fraud': int(is_fraud)
        })
        
    df = pd.DataFrame(data)
    
    os.makedirs('data', exist_ok=True)
    df.to_csv('data/transactions.csv', index=False)
    print(f"Generated {len(df)} transactions. Saved to data/transactions.csv")
    print(f"Fraud Rate: {df.is_fraud.mean():.4f}")

if __name__ == "__main__":
    generate_synthetic_transactions()
