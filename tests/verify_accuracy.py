# tests/verify_accuracy.py
from data.firebase_client import get_transactions, get_budgets, get_goals
from datetime import datetime, timezone

USER_ID = "S6pwTrQB8R7GyuvBdyp0"

# ── Vérifie Budget Agent ──────────────────────────────────────
print("\n── VÉRIFICATION BUDGET AGENT ────────────")
budgets = get_budgets(USER_ID)
transactions = get_transactions(USER_ID, limit=500)

april_expenses = [
    t for t in transactions
    if t.get('type') == 'expense'
    and hasattr(t.get('date'), 'month')
    and t['date'].month == 4 and t['date'].year == 2026
]

spending = {}
for t in april_expenses:
    cat = t.get('category', 'Other')
    spending[cat] = spending.get(cat, 0) + t.get('amount', 0)

print(f"Dépenses avril par catégorie : {spending}")
for b in budgets:
    cat = b.get('category', '')
    limit = b.get('limit_amount', 0)
    spent = spending.get(cat, spending.get(cat.capitalize(), 0))
    pct = round(spent / limit * 100, 1) if limit > 0 else 0
    print(f"  {cat}: {spent:.2f} / {limit:.2f} MAD = {pct}%")

# ── Vérifie Goal Agent ────────────────────────────────────────
print("\n── VÉRIFICATION GOAL AGENT ──────────────")
goals = get_goals(USER_ID)
for g in goals:
    print(f"  {g['title']}: {g['current_amount']}/{g['target_amount']} MAD = {g['progress_pct']}%")

# ── Vérifie Behavioral Agent ──────────────────────────────────
print("\n── VÉRIFICATION BEHAVIORAL AGENT ────────")
from collections import Counter
merchants = [t.get('merchant_name', '') for t in transactions if t.get('type') == 'expense']
top = Counter(merchants).most_common(3)
print(f"Top marchands : {top}")

food_total = sum(t.get('amount', 0) for t in transactions if t.get('category') == 'Food' and t.get('type') == 'expense')
total_expenses = sum(t.get('amount', 0) for t in transactions if t.get('type') == 'expense')
print(f"Food % du total : {round(food_total/total_expenses*100, 1)}%")