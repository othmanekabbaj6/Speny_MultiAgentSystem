from data.firebase_client import (
    get_user, get_transactions, get_budgets, get_goals, get_categories
)

USER_ID = "S6pwTrQB8R7GyuvBdyp0"  # ton vrai user_id

print("\n── USER ──────────────────────────────")
user = get_user(USER_ID)
print(f"  {user['display_name']} | {user['email']} | {user['currency']}")

print("\n── TRANSACTIONS (5 dernières) ────────")
transactions = get_transactions(USER_ID, limit=5)
for t in transactions:
    print(f"  {t['date'].strftime('%Y-%m-%d')} | {t['category']} | {t['amount']} {t['currency']}")

print("\n── BUDGETS ───────────────────────────")
budgets = get_budgets(USER_ID)
for b in budgets:
    print(f"  {b['category']} | limite: {b['limit_amount']} | période: {b['period']}")

print("\n── GOALS ─────────────────────────────")
goals = get_goals(USER_ID)
for g in goals:
    print(f"  {g['title']} | {g['current_amount']}/{g['target_amount']} ({g['progress_pct']}%)")

print("\n── CATEGORIES ────────────────────────")
categories = get_categories(USER_ID)
for c in categories:
    print(f"  {c['name']} | {c['type']}")

print("\n✅ Firebase Client OK !")