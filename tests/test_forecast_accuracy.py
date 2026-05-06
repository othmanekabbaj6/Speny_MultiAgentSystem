from data.firebase_client import get_transactions
from agents.forecast_agent import prepare_prophet_data
import pandas as pd

USER_ID = "S6pwTrQB8R7GyuvBdyp0"

txs = get_transactions(USER_ID, limit=500)

for cat in ['Food', 'Bills', 'Shopping', 'Transport', 'Rental']:
    df = prepare_prophet_data(txs, cat)
    if df.empty:
        print(f"\n{cat} : pas assez de données")
        continue
    print(f"\n=== {cat} ===")
    print(df.to_string(index=False))
    print(f"Moyenne : {df['y'].mean():.2f} MAD")
    print(f"Dernier mois : {df['y'].iloc[-1]:.2f} MAD")
    print(f"CV : {df['y'].std()/df['y'].mean():.2f}")