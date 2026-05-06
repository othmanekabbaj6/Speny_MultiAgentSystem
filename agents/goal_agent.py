from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from data.firebase_client import get_goals, get_transactions
from rag.query_engine import search_user_data
from config.settings import settings
from datetime import datetime, timezone
import logging
logger = logging.getLogger(__name__)

llm = ChatGroq(
    api_key=settings.groq_api_key,
    model_name=settings.llm_model
)

goal_prompt = PromptTemplate(
    input_variables=["goals", "income", "expenses", "context", "question"],
    template="""
Tu es un conseiller financier expert. Analyse les objectifs financiers suivants et réponds en français.

OBJECTIFS FINANCIERS :
{goals}

REVENUS DU MOIS :
{income}

DÉPENSES DU MOIS :
{expenses}

CONTEXTE SUPPLÉMENTAIRE :
{context}

QUESTION : {question}

Réponds de manière encourageante mais réaliste.
Pour chaque objectif, estime le temps restant pour l'atteindre si possible.
Propose des actions concrètes pour accélérer la progression.
"""
)

# ─── Helpers ──────────────────────────────────────────────────
def get_monthly_summary(user_id: str, month: int = None, year: int = None) -> dict:
    """Calcule revenus et dépenses d'un mois donné."""
    now = datetime.now(timezone.utc)
    target_month = month or now.month
    target_year = year or now.year

    all_txs = get_transactions(user_id, limit=500)
    month_txs = []
    for t in all_txs:
        date = t.get('date')
        if date is None:
            continue
        if hasattr(date, 'tzinfo') and date.tzinfo is None:
            date = date.replace(tzinfo=timezone.utc)
        try:
            if date.month == target_month and date.year == target_year:
                month_txs.append(t)
        except AttributeError:
            continue

    income = sum(t.get('amount', 0) for t in month_txs if t.get('type') == 'income')
    expenses = sum(t.get('amount', 0) for t in month_txs if t.get('type') == 'expense')
    return {
        "income": round(income, 2),
        "expenses": round(expenses, 2),
        "savings": round(income - expenses, 2)
    }

def format_goals(goals: list[dict]) -> str:
    """Formate les objectifs avec progression."""
    lines = []
    for g in goals:
        current = g.get('current_amount', 0)
        target = g.get('target_amount', 0)
        pct = g.get('progress_pct', 0)
        remaining = round(target - current, 2)

        if pct >= 100:
            status = "✅ ATTEINT"
        elif pct >= 75:
            status = "🟢 Presque là"
        elif pct >= 40:
            status = "🟡 En cours"
        else:
            status = "🔴 Début"

        lines.append(
            f"- {g.get('title')} : {current}/{target} MAD "
            f"({pct}%) [{status}] — Reste : {remaining} MAD"
        )
    return "\n".join(lines) if lines else "Aucun objectif défini."

# ─── Agent principal ──────────────────────────────────────────
def run_goal_agent(user_id: str, question: str = None, month: int = None, year: int = None) -> dict:
    if question is None:
        question = "Analyse mes objectifs financiers. Suis-je sur la bonne voie ?"

    goals = get_goals(user_id)
    summary = get_monthly_summary(user_id, month=month, year=year)

    context_docs = search_user_data(user_id, question, top_k=5)
    context = "\n".join([d['text'] for d in context_docs])

    chain = goal_prompt | llm
    response = chain.invoke({
        "goals": format_goals(goals),
        "income": f"{summary['income']} MAD",
        "expenses": f"{summary['expenses']} MAD",
        "context": context,
        "question": question
    })

    return {
        "agent": "goal_agent",
        "question": question,
        "answer": response.content,
        "goals": goals,
        "monthly_summary": summary
    }