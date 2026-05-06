from rag.indexer import build_user_index
from rag.query_engine import search_user_data

USER_ID = "S6pwTrQB8R7GyuvBdyp0"

print("\n── Indexation des données ────────────")
index = build_user_index(USER_ID)
print("✅ Index créé !")

print("\n── Test recherche sémantique ──────────")
queries = [
    "dépenses Transport ce mois-ci",
    "objectif voyage Japon",
    "budget food dépassé",
]

for query in queries:
    print(f"\n🔍 Query: '{query}'")
    results = search_user_data(USER_ID, query, top_k=3)
    for r in results:
        print(f"  [{r['score']}] {r['text'][:100]}...")