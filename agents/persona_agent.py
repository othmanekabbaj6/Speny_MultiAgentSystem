import pandas as pd
import numpy as np
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from data.firebase_client import get_transactions, get_budgets, get_goals, get_user
from agents.budget_agent import get_current_month_transactions, calculate_spending_by_category
from agents.behavioral_agent import (
    analyze_categories, analyze_day_of_week,
    analyze_purchase_frequency, analyze_monthly_trends
)
from rag.query_engine import search_user_data
from config.settings import settings
from datetime import datetime, timezone
import logging

logger = logging.getLogger(__name__)

llm = ChatGroq(
    api_key=settings.groq_api_key,
    model_name=settings.llm_model
)

persona_prompt = PromptTemplate(
    input_variables=["profile", "context", "question"],
    template="""
Tu es un expert en psychologie financière et coaching financier.
Analyse le profil financier suivant et réponds en français.

PROFIL FINANCIER DE L'UTILISATEUR :
{profile}

CONTEXTE SUPPLÉMENTAIRE :
{context}

QUESTION : {question}

Sur la base de ce profil :
1. Définis le type financier principal de l'utilisateur
   (Épargnant / Dépensier / Impulsif / Équilibré / Investisseur)
2. Explique les forces financières de l'utilisateur
3. Explique les points d'amélioration prioritaires
4. Donne 3 recommandations personnalisées et concrètes
   adaptées à ce profil spécifique

Sois précis, bienveillant et personnalisé.
Évite les conseils génériques — base-toi sur les données réelles.
"""
)

# ─── Score de discipline budgétaire (0-100) ───────────────────
def calculate_budget_discipline_score(
    budgets: list[dict],
    transactions: list[dict]
) -> dict:
    """
    Calcule un score de discipline budgétaire basé sur
    le respect des budgets définis.
    Score 100 = parfaitement dans les budgets
    Score 0   = tous les budgets dépassés
    """
    if not budgets:
        return {'score': 50, 'label': 'Non évalué', 'details': []}

    # Dépenses des 3 derniers mois
    now = datetime.now(timezone.utc)
    recent_txs = []
    for t in transactions:
        date = t.get('date')
        if hasattr(date, 'tzinfo') and date.tzinfo is None:
            date = date.replace(tzinfo=timezone.utc)
        if date and (now - date).days <= 90:
            recent_txs.append(t)

    expenses = [t for t in recent_txs if t.get('type') == 'expense']

    # Calcule dépenses par catégorie par mois
    monthly_spending = {}
    for t in expenses:
        date = t.get('date')
        if hasattr(date, 'tzinfo') and date.tzinfo is None:
            date = date.replace(tzinfo=timezone.utc)
        key = f"{date.year}-{date.month:02d}"
        cat = t.get('category', '').lower()
        if key not in monthly_spending:
            monthly_spending[key] = {}
        monthly_spending[key][cat] = monthly_spending[key].get(cat, 0) + t.get('amount', 0)

    # Score par budget
    details = []
    scores = []
    for b in budgets:
        cat = b.get('category', '').lower()
        limit = b.get('limit_amount', 0)
        if limit == 0:
            continue

        month_scores = []
        for month, spending in monthly_spending.items():
            spent = spending.get(cat, 0)
            ratio = spent / limit
            month_score = max(0, 100 - max(0, (ratio - 1) * 100))
            month_scores.append(month_score)

        avg_score = np.mean(month_scores) if month_scores else 50
        scores.append(avg_score)
        details.append({
            'category': b.get('category'),
            'score': round(avg_score, 1),
            'limit': limit
        })

    final_score = round(np.mean(scores), 1) if scores else 50

    if final_score >= 80:
        label = "Excellent"
    elif final_score >= 60:
        label = "Bien"
    elif final_score >= 40:
        label = "À améliorer"
    else:
        label = "Critique"

    return {
        'score': final_score,
        'label': label,
        'details': details
    }

# ─── Score d'épargne (0-100) ──────────────────────────────────
def calculate_savings_score(transactions: list[dict]) -> dict:
    """
    Calcule un score d'épargne basé sur le ratio épargne/revenus.
    """
    now = datetime.now(timezone.utc)
    recent_txs = []
    for t in transactions:
        date = t.get('date')
        if hasattr(date, 'tzinfo') and date.tzinfo is None:
            date = date.replace(tzinfo=timezone.utc)
        if date and (now - date).days <= 90:
            recent_txs.append(t)

    income = sum(t.get('amount', 0) for t in recent_txs if t.get('type') == 'income')
    expenses = sum(t.get('amount', 0) for t in recent_txs if t.get('type') == 'expense')

    if income == 0:
        return {'score': 0, 'ratio': 0, 'label': 'Pas de revenus détectés'}

    savings_ratio = (income - expenses) / income * 100
    score = min(100, max(0, savings_ratio * 2))

    if savings_ratio >= 20:
        label = "Excellent épargnant (>20%)"
    elif savings_ratio >= 10:
        label = "Bon épargnant (10-20%)"
    elif savings_ratio >= 0:
        label = "Épargnant modéré (0-10%)"
    else:
        label = "Dépenses supérieures aux revenus"

    return {
        'score': round(score, 1),
        'ratio': round(savings_ratio, 1),
        'income': round(income, 2),
        'expenses': round(expenses, 2),
        'label': label
    }

# ─── Score de régularité (0-100) ──────────────────────────────
def calculate_regularity_score(transactions: list[dict]) -> dict:
    """
    Mesure la régularité des dépenses mois par mois.
    Une faible volatilité = score élevé.
    """
    df = pd.DataFrame([{
        'amount': t.get('amount', 0),
        'type': t.get('type', ''),
        'date': t.get('date')
    } for t in transactions if t.get('type') == 'expense'])

    if len(df) < 10:
        return {'score': 50, 'label': 'Pas assez de données'}

    df['month'] = df['date'].apply(
        lambda d: f"{d.year}-{d.month:02d}" if hasattr(d, 'year') else 'unknown'
    )

    monthly = df.groupby('month')['amount'].sum()
    if len(monthly) < 2:
        return {'score': 50, 'label': 'Pas assez de mois'}

    cv = monthly.std() / monthly.mean() * 100
    score = max(0, 100 - cv)

    if score >= 80:
        label = "Très régulier"
    elif score >= 60:
        label = "Assez régulier"
    elif score >= 40:
        label = "Irrégulier"
    else:
        label = "Très irrégulier"

    return {
        'score': round(score, 1),
        'volatility': round(cv, 1),
        'label': label
    }

# ─── Score de progression des goals (0-100) ───────────────────
def calculate_goals_score(goals: list[dict]) -> dict:
    """
    Score basé sur la progression moyenne des objectifs actifs.
    """
    active_goals = [g for g in goals if g.get('status') == 'active']

    if not active_goals:
        return {'score': 0, 'label': 'Aucun objectif défini'}

    avg_progress = np.mean([g.get('progress_pct', 0) for g in active_goals])
    score = min(100, avg_progress)

    if score >= 75:
        label = "Excellent"
    elif score >= 50:
        label = "En bonne voie"
    elif score >= 25:
        label = "Début"
    else:
        label = "À démarrer"

    return {
        'score': round(score, 1),
        'avg_progress': round(avg_progress, 1),
        'active_goals_count': len(active_goals),
        'label': label
    }

# ─── Classification du type financier ─────────────────────────
def classify_financial_type(
    budget_score: float,
    savings_score: float,
    regularity_score: float,
    goals_score: float,
    savings_ratio: float
) -> dict:
    """
    Classifie le profil financier en fonction des scores.
    """
    global_score = round(
        budget_score * 0.3 +
        savings_score * 0.3 +
        regularity_score * 0.2 +
        goals_score * 0.2,
        1
    )

    if savings_ratio >= 20 and budget_score >= 70:
        persona = "💎 Épargnant discipliné"
        description = "Tu épargnes significativement et respectes tes budgets."
    elif savings_ratio >= 10 and regularity_score >= 60:
        persona = "⚖️ Profil équilibré"
        description = "Tu as de bonnes habitudes financières avec une épargne régulière."
    elif budget_score < 40:
        persona = "🔴 Dépensier impulsif"
        description = "Tu as tendance à dépasser tes budgets régulièrement."
    elif regularity_score < 40:
        persona = "🌊 Profil irrégulier"
        description = "Tes dépenses sont très variables d'un mois à l'autre."
    elif goals_score >= 60:
        persona = "🎯 Orienté objectifs"
        description = "Tu progresses bien vers tes objectifs financiers."
    else:
        persona = "📈 Profil en développement"
        description = "Tu as des bases solides mais des améliorations sont possibles."

    risk_tolerance = "Faible" if savings_ratio < 5 else \
                     "Modérée" if savings_ratio < 15 else "Élevée"

    return {
        'persona': persona,
        'description': description,
        'global_score': global_score,
        'risk_tolerance': risk_tolerance
    }

# ─── Formatage du profil ──────────────────────────────────────
def format_profile(
    user: dict,
    budget_discipline: dict,
    savings: dict,
    regularity: dict,
    goals_score: dict,
    financial_type: dict,
    cat_analysis: dict,
    trend_analysis: dict
) -> str:
    lines = []

    lines.append("👤 IDENTITÉ FINANCIÈRE :")
    lines.append(f"  - Type : {financial_type['persona']}")
    lines.append(f"  - {financial_type['description']}")
    lines.append(f"  - Score global : {financial_type['global_score']}/100")
    lines.append(f"  - Tolérance au risque : {financial_type['risk_tolerance']}")

    lines.append("\n💰 ÉPARGNE :")
    lines.append(f"  - Taux d'épargne : {savings['ratio']}%")
    lines.append(f"  - Score épargne : {savings['score']}/100 — {savings['label']}")
    lines.append(f"  - Revenus (3 mois) : {savings['income']} MAD")
    lines.append(f"  - Dépenses (3 mois) : {savings['expenses']} MAD")

    lines.append("\n📊 DISCIPLINE BUDGÉTAIRE :")
    lines.append(
        f"  - Score : {budget_discipline['score']}/100 — {budget_discipline['label']}"
    )
    for d in budget_discipline['details']:
        lines.append(
            f"  - {d['category']} : {d['score']}/100 "
            f"(limite {d['limit']} MAD)"
        )

    lines.append("\n📅 RÉGULARITÉ :")
    lines.append(
        f"  - Score : {regularity['score']}/100 — {regularity['label']}"
    )
    lines.append(f"  - Volatilité mensuelle : {regularity.get('volatility', 0)}%")

    lines.append("\n🎯 OBJECTIFS :")
    lines.append(
        f"  - Score : {goals_score['score']}/100 — {goals_score['label']}"
    )
    lines.append(
        f"  - Progression moyenne : {goals_score.get('avg_progress', 0)}% "
        f"sur {goals_score.get('active_goals_count', 0)} objectifs actifs"
    )

    lines.append("\n🏷️ RÉPARTITION DES DÉPENSES :")
    cats = cat_analysis['categories']['pct']
    for cat, pct in sorted(cats.items(), key=lambda x: -x[1])[:5]:
        lines.append(f"  - {cat} : {pct}%")

    if trend_analysis.get('trend_direction'):
        lines.append("\n📈 TENDANCE :")
        lines.append(
            f"  - {trend_analysis['trend_direction']} de "
            f"{abs(trend_analysis['trend_pct'])}% sur la période"
        )

    return "\n".join(lines)

# ─── Agent principal ───────────────────────────────────────────
def run_persona_agent(user_id: str, question: str = None) -> dict:
    """
    Construit le profil financier dynamique de l'utilisateur.
    """
    if question is None:
        question = "Quel est mon profil financier et comment puis-je l'améliorer ?"

    # 1. Récupère toutes les données
    user = get_user(user_id) or {}
    transactions = get_transactions(user_id, limit=500)
    budgets = get_budgets(user_id)
    goals = get_goals(user_id)
    expenses = [t for t in transactions if t.get('type') == 'expense']

    # 2. Calcule tous les scores
    budget_discipline = calculate_budget_discipline_score(budgets, transactions)
    savings = calculate_savings_score(transactions)
    regularity = calculate_regularity_score(transactions)
    goals_score = calculate_goals_score(goals)

    # 3. Classification du type financier
    financial_type = classify_financial_type(
        budget_score=budget_discipline['score'],
        savings_score=savings['score'],
        regularity_score=regularity['score'],
        goals_score=goals_score['score'],
        savings_ratio=savings.get('ratio', 0)
    )

    # 4. Analyses comportementales
    df = pd.DataFrame([{
        'amount': t.get('amount', 0),
        'category': t.get('category', 'Other'),
        'date': t.get('date'),
        'type': t.get('type', '')
    } for t in expenses])

    def normalize_date(d):
        if d is None:
            return None
        if hasattr(d, 'tzinfo') and d.tzinfo is None:
            return d.replace(tzinfo=timezone.utc)
        return d

    df['date'] = df['date'].apply(normalize_date)

    cat_analysis = analyze_categories(df.copy())
    trend_analysis = analyze_monthly_trends(df.copy())

    # 5. Contexte RAG
    context_docs = search_user_data(user_id, question, top_k=5)
    context = "\n".join([d['text'] for d in context_docs])

    # 6. Formate le profil
    profile_str = format_profile(
        user, budget_discipline, savings,
        regularity, goals_score, financial_type,
        cat_analysis, trend_analysis
    )

    # 7. Appel LLM
    chain = persona_prompt | llm
    response = chain.invoke({
        "profile": profile_str,
        "context": context,
        "question": question
    })

    return {
        "agent": "persona_agent",
        "question": question,
        "answer": response.content,
        "persona": financial_type,
        "scores": {
            "budget_discipline": budget_discipline['score'],
            "savings": savings['score'],
            "regularity": regularity['score'],
            "goals": goals_score['score'],
            "global": financial_type['global_score']
        },
        "savings_ratio": savings.get('ratio', 0),
        "risk_tolerance": financial_type['risk_tolerance']
    }