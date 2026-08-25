import os
import pandas as pd
import numpy as np

DATA_DIR = "data"

def load_data(data_dir=DATA_DIR):
    """Loads raw data or generates synthetic sample data if none exists."""
    os.makedirs(data_dir, exist_ok=True)
    file_path = os.path.join(data_dir, "online_retail.csv")
    
    if os.path.exists(file_path):
        df = pd.read_csv(file_path)
    else:
        dates = pd.date_range(start="2017-11-01", periods=500, freq="D")
        np.random.seed(42)
        
        countries = ["United Kingdom", "Germany", "EIRE", "France"]
        records = []
        
        for d in dates:
            for c in countries:
                records.append({
                    "invoice": f"INV-{np.random.randint(10000, 99999)}",
                    "date": d,
                    "country": c,
                    "price": round(np.random.uniform(10.0, 500.0), 2),
                    "purchases": np.random.randint(1, 10)
                })
        df = pd.DataFrame(records)
        df.to_csv(file_path, index=False)
        
    df["date"] = pd.to_datetime(df["date"])
    return df

def aggregate_daily(df, country=None):
    """Aggregates transactional data into daily total revenue."""
    df_filtered = df.copy()
    
    if country and country.lower() != "all":
        df_filtered = df_filtered[df_filtered["country"].str.lower() == country.lower()]
        
    daily_df = df_filtered.groupby("date").agg({
        "price": "sum"
    }).reset_index()
    
    daily_df.rename(columns={"price": "revenue"}, inplace=True)
    daily_df = daily_df.sort_values("date").reset_index(drop=True)
    return daily_df

if __name__ == "__main__":
    df = load_data()
    daily_uk = aggregate_daily(df, country="United Kingdom")
    print(f"Data Ingestion Complete. Aggregated rows: {len(daily_uk)}")