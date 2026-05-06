from data.firebase_client import get_transactions
from agents.anomaly_agent import (
    prepare_features, encode_features,
    get_recurring_merchants, get_monthly_category_stats,
    calculate_deviation, classify_anomaly,
    detect_anomalies_global
)
import pandas as pd

USER_ID = "S6pwTrQB8R7GyuvBdyp0"

txs = get_transactions(USER_ID, limit=500)
expenses = [t for t in txs if t.get('type') == 'expense']

df_raw = pd.DataFrame([{
    'category': t.get('category'),
    'amount': t.get('amount'),
    'merchant': t.get('merchant_name'),
    'date': t['date'].strftime('%Y-%m-%d') if hasattr(t['date'], 'strftime') else ''
} for t in expenses])

# ── Niveau 1 : Stats Amazon ────────────────────────────────────
print("\n=== AMAZON ===")
amazon = df_raw[df_raw['merchant'].str.lower().str.contains('amazon', na=False)]
print(amazon[['date','amount']].sort_values('date').to_string(index=False))
print(f"\nMoyenne : {amazon['amount'].mean():.2f} MAD")
print(f"Max     : {amazon['amount'].max():.2f} MAD")
print(f"Min     : {amazon['amount'].min():.2f} MAD")

# ── Niveau 2 : Stats par catégorie ────────────────────────────
print("\n=== STATS PAR CATÉGORIE ===")
stats = df_raw.groupby('category')['amount'].agg(['mean','std','min','max','count'])
print(stats.round(2).to_string())

# ── Niveau 3 : Marchands récurrents détectés ──────────────────
print("\n=== MARCHANDS RÉCURRENTS (exclus de l'analyse) ===")
recurring = get_recurring_merchants(expenses, threshold=6)
for m in sorted(recurring):
    count = sum(1 for t in expenses if t.get('merchant_name','').lower().strip() == m)
    print(f"  {m} → {count} occurrences")

# ── Niveau 4 : Anomalies via l'agent réel ────────────────────
print("\n=== ANOMALIES APRÈS FILTRAGE (ce que l'agent voit réellement) ===")

filtered = [
    t for t in expenses
    if t.get('merchant_name', '').lower().strip() not in recurring
]
print(f"Transactions avant filtrage : {len(expenses)}")
print(f"Transactions après filtrage : {len(filtered)}")

# Utilise directement detect_anomalies_global qui contient tous les filtres
anomalies = detect_anomalies_global(expenses)

for a in anomalies:
    print(
        f"  {a['anomaly_type']:<35} | {a['date']} | "
        f"{a['merchant']:<25} | {a['category']:<15} | "
        f"{a['amount']:.2f} MAD | "
        f"moy({a['context_label']}): {a['monthly_mean']:.2f} | "
        f"dév: {a['deviation_from_mean']}x"
    )

print(f"\nTotal anomalies après filtrage : {len(anomalies)}")
print(f"Total transactions analysées   : {len(filtered)}")
print(f"Seuil contamination 5%         : ~{int(len(filtered)*0.05)} transactions attendues")

# ── Niveau 5 : Validation manuelle ───────────────────────────
print("\n=== VALIDATION MANUELLE ===")
print("Questions à se poser :")
print(
    "  ✅ Monthly Rent apparaît-il encore ?",
    any('rent' in a['merchant'].lower() for a in anomalies)
)
print(
    "  ✅ Electroplanet détecté ?",
    any('electroplanet' in a['merchant'].lower() for a in anomalies)
)
print(
    "  ✅ Amazon 1862 MAD détecté ?",
    any('amazon' in a['merchant'].lower() for a in anomalies)
)
print(
    "  ✅ Lydec absent ?",
    not any('lydec' in a['merchant'].lower() for a in anomalies)
)
print(
    "  ✅ Netflix absent ?",
    not any('netflix' in a['merchant'].lower() for a in anomalies)
)