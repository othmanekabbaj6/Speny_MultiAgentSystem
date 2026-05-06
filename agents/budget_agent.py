from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from data.firebase_client import get_transactions, get_budgets
from rag.query_engine import search_user_data
from config.settings import settings
from datetime import datetime, timezone
import logging

logger = logging.getLogger(__name__)

# ─── LLM ──────────────────────────────────────────────────────
llm = ChatGroq(
    api_key=settings.groq_api_key,
    model_name=settings.llm_model
)

# ─── Prompt ───────────────────────────────────────────────────
budget_prompt = PromptTemplate(
    input_variables=["budgets", "transactions", "spending_no_budget", "context", "question"],
    template="""
Tu es un assistant financier expert. Analyse les données suivantes et réponds en français.

BUDGETS DÉFINIS PAR L'UTILISATEUR :
{budgets}

DÉPENSES DANS DES CATÉGORIES SANS BUDGET :
{spending_no_budget}

TRANSACTIONS RÉCENTES :
{transactions}

CONTEXTE SUPPLÉMENTAIRE :
{context}

QUESTION : {question}

Règles importantes :
- Ne parle que des budgets qui sont explicitement définis
- Mentionne séparément les catégories dépensées mais sans budget défini
- Propose à l'utilisateur de créer des budgets pour ces catégories
- Sois clair, concis et actionnable
"""
)

# ─── Helpers ──────────────────────────────────────────────────
def get_current_month_transactions(user_id: str, month: int = None, year: int = None) -> list[dict]:
    """Filtre les transactions d'un mois donné (par défaut : mois en cours)."""
    now = datetime.now(timezone.utc)
    target_month = month or now.month
    target_year = year or now.year

    all_txs = get_transactions(user_id, limit=500)
    result = []
    for t in all_txs:
        date = t.get('date')
        if date is None:
            continue
        if hasattr(date, 'tzinfo') and date.tzinfo is None:
            date = date.replace(tzinfo=timezone.utc)
        try:
            if date.month == target_month and date.year == target_year:
                result.append(t)
        except AttributeError:
            continue
    return result

def calculate_spending_by_category(transactions: list[dict]) -> dict:
    """Calcule le total dépensé par catégorie."""
    spending = {}
    for t in transactions:
        if t.get('type') == 'expense':
            cat = t.get('category', 'Autre')
            spending[cat] = spending.get(cat, 0) + t.get('amount', 0)
    return spending

def format_budgets(budgets: list[dict], spending: dict) -> str:
    """Formate les budgets avec le statut actuel."""
    lines = []
    for b in budgets:
        cat = b.get('category', '')
        limit = b.get('limit_amount', 0)
        spent = spending.get(cat, spending.get(cat.capitalize(), 0))
        pct = round(spent / limit * 100, 1) if limit > 0 else 0
        alert_at = int(b.get('alert_at', 0.8) * 100)

        if pct >= 100:
            status = "🔴 DÉPASSÉ"
        elif pct >= alert_at:
            status = "🟠 ALERTE"
        else:
            status = "🟢 OK"

        lines.append(
            f"- {cat.capitalize()} : {spent:.2f}/{limit:.2f} MAD "
            f"({pct}%) [{status}]"
        )
    return "\n".join(lines) if lines else "Aucun budget défini."

def format_transactions(transactions: list[dict], limit: int = 10) -> str:
    """Formate les transactions pour le prompt."""
    lines = []
    for t in transactions[:limit]:
        date_str = t['date'].strftime('%Y-%m-%d') if hasattr(t['date'], 'strftime') else str(t['date'])
        lines.append(
            f"- {date_str} | {t.get('category')} | "
            f"{t.get('merchant_name', '?')} | "
            f"{t.get('amount', 0):.2f} MAD | {t.get('type')}"
        )
    return "\n".join(lines) if lines else "Aucune transaction ce mois-ci."

# ─── Agent principal ──────────────────────────────────────────
def run_budget_agent(user_id: str, question: str = None, month: int = None, year: int = None) -> dict:
    if question is None:
        question = "Analyse mon budget du mois en cours. Y a-t-il des dépassements ou des alertes ?"

    budgets = get_budgets(user_id)
    month_transactions = get_current_month_transactions(user_id, month=month, year=year)
    spending = calculate_spending_by_category(month_transactions)

    # Catégories avec budget
    budgeted_categories = [b.get('category', '').lower() for b in budgets]

    # Dépenses sans budget défini
    spending_no_budget = {
        cat: amount for cat, amount in spending.items()
        if cat.lower() not in budgeted_categories
    }

    # Contexte RAG
    context_docs = search_user_data(user_id, question, top_k=5)
    context = "\n".join([d['text'] for d in context_docs])

    budgets_str = format_budgets(budgets, spending)
    transactions_str = format_transactions(month_transactions)

    # Formate les dépenses sans budget
    no_budget_str = "\n".join([
        f"- {cat} : {amount:.2f} MAD (aucun budget défini)"
        for cat, amount in spending_no_budget.items()
    ]) if spending_no_budget else "Aucune"

    chain = budget_prompt | llm
    response = chain.invoke({
        "budgets": budgets_str,
        "transactions": transactions_str,
        "spending_no_budget": no_budget_str,
        "context": context,
        "question": question
    })

    return {
        "agent": "budget_agent",
        "question": question,
        "answer": response.content,
        "budgets": budgets,
        "spending_by_category": spending,
        "spending_no_budget": spending_no_budget,
        "month_transactions_count": len(month_transactions)
    }