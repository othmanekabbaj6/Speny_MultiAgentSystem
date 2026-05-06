import pandas as pd
import numpy as np
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from data.firebase_client import get_transactions
from rag.query_engine import search_user_data
from config.settings import settings
from datetime import datetime, timezone
import logging

logger = logging.getLogger(__name__)

llm = ChatGroq(
    api_key=settings.groq_api_key,
    model_name=settings.llm_model
)

behavioral_prompt = PromptTemplate(
    input_variables=["patterns", "context", "question"],
    template="""
Tu es un coach financier bienveillant et perspicace.
Analyse les habitudes de dépenses suivantes et réponds en français.

HABITUDES DE DÉPENSES DÉTECTÉES :
{patterns}

CONTEXTE SUPPLÉMENTAIRE :
{context}

QUESTION : {question}

Pour chaque habitude identifiée :
- Décris le pattern de manière claire et concrète
- Explique l'impact financier sur le long terme
- Propose 1 action concrète et réaliste pour améliorer cette habitude

Sois encourageant, précis et actionnable. Évite les jugements négatifs.
Commence par les habitudes les plus impactantes financièrement.
"""
)

# ─── Analyse par jour de la semaine ───────────────────────────
def analyze_day_of_week(df: pd.DataFrame) -> dict:
    """Détecte les jours où l'utilisateur dépense le plus."""
    days_fr = {
        0: 'Lundi', 1: 'Mardi', 2: 'Mercredi',
        3: 'Jeudi', 4: 'Vendredi', 5: 'Samedi', 6: 'Dimanche'
    }

    df['day_of_week'] = df['date'].apply(
        lambda d: d.weekday() if hasattr(d, 'weekday') else 0
    )
    df['day_name'] = df['day_of_week'].map(days_fr)

    daily_spending = df.groupby('day_name')['amount'].agg(['sum', 'count', 'mean'])
    daily_spending = daily_spending.reindex(list(days_fr.values()))

    top_day = daily_spending['sum'].idxmax()
    top_day_total = daily_spending.loc[top_day, 'sum']
    top_day_count = daily_spending.loc[top_day, 'count']

    weekend_total = daily_spending.loc[['Samedi', 'Dimanche'], 'sum'].sum()
    weekday_total = daily_spending.loc[
        ['Lundi', 'Mardi', 'Mercredi', 'Jeudi', 'Vendredi'], 'sum'
    ].sum()
    weekend_pct = round(weekend_total / (weekend_total + weekday_total) * 100, 1)

    return {
        'top_day': top_day,
        'top_day_total': round(top_day_total, 2),
        'top_day_count': int(top_day_count),
        'weekend_pct': weekend_pct,
        'weekend_total': round(weekend_total, 2),
        'weekday_total': round(weekday_total, 2),
        'daily_breakdown': daily_spending.round(2).to_dict()
    }

# ─── Analyse par catégorie ────────────────────────────────────
def analyze_categories(df: pd.DataFrame) -> dict:
    """Identifie les catégories dominantes et leur évolution."""
    cat_stats = df.groupby('category')['amount'].agg(['sum', 'count', 'mean'])
    total = cat_stats['sum'].sum()
    cat_stats['pct'] = (cat_stats['sum'] / total * 100).round(1)
    cat_stats = cat_stats.sort_values('sum', ascending=False)

    top_category = cat_stats.index[0]
    top_pct = cat_stats.loc[top_category, 'pct']

    return {
        'top_category': top_category,
        'top_category_pct': top_pct,
        'top_category_total': round(cat_stats.loc[top_category, 'sum'], 2),
        'categories': cat_stats.round(2).to_dict(),
        'total_expenses': round(total, 2)
    }

# ─── Analyse de la fréquence d'achat ─────────────────────────
def analyze_purchase_frequency(df: pd.DataFrame) -> dict:
    """Détecte les habitudes d'achat fréquentes et impulsives."""
    df['month_year'] = df['date'].apply(
        lambda d: f"{d.year}-{d.month:02d}" if hasattr(d, 'year') else 'unknown'
    )

    monthly_counts = df.groupby('month_year')['amount'].count()
    avg_transactions_per_month = round(monthly_counts.mean(), 1)
    max_transactions_month = monthly_counts.idxmax()
    max_transactions_count = int(monthly_counts.max())

    # Petites dépenses fréquentes (< 100 MAD)
    small_purchases = df[df['amount'] < 100]
    small_pct = round(len(small_purchases) / len(df) * 100, 1)
    small_total = round(small_purchases['amount'].sum(), 2)

    # Grandes dépenses (> 1000 MAD)
    large_purchases = df[df['amount'] > 1000]
    large_count = len(large_purchases)
    large_total = round(large_purchases['amount'].sum(), 2)

    return {
        'avg_transactions_per_month': avg_transactions_per_month,
        'max_transactions_month': max_transactions_month,
        'max_transactions_count': max_transactions_count,
        'small_purchases_pct': small_pct,
        'small_purchases_total': small_total,
        'large_purchases_count': large_count,
        'large_purchases_total': large_total,
    }

# ─── Analyse de la régularité mensuelle ───────────────────────
def analyze_monthly_trends(df: pd.DataFrame) -> dict:
    """Analyse l'évolution des dépenses mois par mois."""
    df['month_year'] = df['date'].apply(
        lambda d: f"{d.year}-{d.month:02d}" if hasattr(d, 'year') else 'unknown'
    )

    monthly_totals = df.groupby('month_year')['amount'].sum().sort_index()

    if len(monthly_totals) < 2:
        return {'trend': 'insufficient_data'}

    # Tendance générale (hausse ou baisse)
    first_half = monthly_totals.iloc[:len(monthly_totals)//2].mean()
    second_half = monthly_totals.iloc[len(monthly_totals)//2:].mean()
    trend_pct = round((second_half - first_half) / first_half * 100, 1)

    # Mois le plus dépensier
    top_month = monthly_totals.idxmax()
    top_month_total = round(monthly_totals.max(), 2)

    # Mois le moins dépensier
    low_month = monthly_totals.idxmin()
    low_month_total = round(monthly_totals.min(), 2)

    # Volatilité (coefficient de variation)
    volatility = round(monthly_totals.std() / monthly_totals.mean() * 100, 1)

    return {
        'trend_pct': trend_pct,
        'trend_direction': 'hausse' if trend_pct > 5 else 'baisse' if trend_pct < -5 else 'stable',
        'top_month': top_month,
        'top_month_total': top_month_total,
        'low_month': low_month,
        'low_month_total': low_month_total,
        'volatility_pct': volatility,
        'monthly_totals': monthly_totals.round(2).to_dict()
    }

# ─── Analyse des habitudes de restauration ───────────────────
def analyze_food_habits(df: pd.DataFrame) -> dict:
    """Analyse spécifique aux dépenses Food."""
    food_df = df[df['category'].str.lower() == 'food']

    if len(food_df) == 0:
        return {'found': False}

    food_total = round(food_df['amount'].sum(), 2)
    food_count = len(food_df)
    food_avg = round(food_df['amount'].mean(), 2)

    total_expenses = df['amount'].sum()
    food_pct = round(food_total / total_expenses * 100, 1)

    # Fréquence par semaine
    food_df = food_df.copy()
    food_df['week'] = food_df['date'].apply(
        lambda d: f"{d.year}-W{d.isocalendar()[1]:02d}" if hasattr(d, 'isocalendar') else 'unknown'
    )
    weekly_food = food_df.groupby('week')['amount'].count()
    avg_food_per_week = round(weekly_food.mean(), 1)

    return {
        'found': True,
        'food_total': food_total,
        'food_count': food_count,
        'food_avg_per_transaction': food_avg,
        'food_pct_of_total': food_pct,
        'avg_food_transactions_per_week': avg_food_per_week,
    }

# ─── Formatage des patterns ───────────────────────────────────
def format_patterns(
    day_analysis: dict,
    cat_analysis: dict,
    freq_analysis: dict,
    trend_analysis: dict,
    food_analysis: dict
) -> str:
    lines = []

    # Jour de la semaine
    lines.append("📅 HABITUDES PAR JOUR :")
    lines.append(
        f"  - Jour le plus dépensier : {day_analysis['top_day']} "
        f"({day_analysis['top_day_total']} MAD, {day_analysis['top_day_count']} transactions)"
    )
    lines.append(
        f"  - Dépenses weekend : {day_analysis['weekend_pct']}% du total "
        f"({day_analysis['weekend_total']} MAD)"
    )

    # Catégories
    lines.append("\n🏷️ CATÉGORIES DOMINANTES :")
    lines.append(
        f"  - Catégorie principale : {cat_analysis['top_category']} "
        f"({cat_analysis['top_category_pct']}% du total, "
        f"{cat_analysis['top_category_total']} MAD)"
    )
    cats = cat_analysis['categories']['pct']
    for cat, pct in sorted(cats.items(), key=lambda x: -x[1])[:5]:
        lines.append(f"  - {cat} : {pct}%")

    # Fréquence
    lines.append("\n🔁 FRÉQUENCE D'ACHAT :")
    lines.append(
        f"  - Moyenne : {freq_analysis['avg_transactions_per_month']} transactions/mois"
    )
    lines.append(
        f"  - Petites dépenses (< 100 MAD) : {freq_analysis['small_purchases_pct']}% "
        f"des transactions ({freq_analysis['small_purchases_total']} MAD au total)"
    )
    lines.append(
        f"  - Grandes dépenses (> 1000 MAD) : {freq_analysis['large_purchases_count']} "
        f"transactions ({freq_analysis['large_purchases_total']} MAD au total)"
    )

    # Tendance mensuelle
    if trend_analysis.get('trend_direction'):
        lines.append("\n📈 TENDANCE MENSUELLE :")
        lines.append(
            f"  - Tendance générale : {trend_analysis['trend_direction']} "
            f"({trend_analysis['trend_pct']}%)"
        )
        lines.append(
            f"  - Mois le plus dépensier : {trend_analysis['top_month']} "
            f"({trend_analysis['top_month_total']} MAD)"
        )
        lines.append(
            f"  - Volatilité mensuelle : {trend_analysis['volatility_pct']}%"
        )

    # Food
    if food_analysis.get('found'):
        lines.append("\n🍽️ HABITUDES ALIMENTAIRES :")
        lines.append(
            f"  - Food = {food_analysis['food_pct_of_total']}% de tes dépenses totales"
        )
        lines.append(
            f"  - {food_analysis['avg_food_transactions_per_week']} "
            f"transactions alimentaires/semaine en moyenne"
        )
        lines.append(
            f"  - Dépense moyenne par repas/course : "
            f"{food_analysis['food_avg_per_transaction']} MAD"
        )

    return "\n".join(lines)

# ─── Agent principal ───────────────────────────────────────────
def run_behavioral_agent(user_id: str, question: str = None) -> dict:
    """
    Analyse les habitudes de dépenses de l'utilisateur.
    """
    if question is None:
        question = "Quelles sont mes habitudes de dépenses ? Y a-t-il des patterns à améliorer ?"

    # 1. Récupère toutes les dépenses
    transactions = get_transactions(user_id, limit=500)
    expenses = [t for t in transactions if t.get('type') == 'expense']

    if len(expenses) == 0:
        return {
            "agent": "behavioral_agent",
            "answer": "Pas assez de données pour analyser les habitudes.",
            "behavior": {}
        }

    # 2. Prépare le DataFrame
    df = pd.DataFrame([{
        'amount': t.get('amount', 0),
        'category': t.get('category', 'Other'),
        'merchant': t.get('merchant_name', ''),
        'date': t.get('date'),
        'type': t.get('type', '')
    } for t in expenses])

    # Normalise les dates
    def normalize_date(d):
        if d is None:
            return None
        if hasattr(d, 'tzinfo') and d.tzinfo is None:
            return d.replace(tzinfo=timezone.utc)
        return d

    df['date'] = df['date'].apply(normalize_date)

    # 3. Lance toutes les analyses
    day_analysis = analyze_day_of_week(df.copy())
    cat_analysis = analyze_categories(df.copy())
    freq_analysis = analyze_purchase_frequency(df.copy())
    trend_analysis = analyze_monthly_trends(df.copy())
    food_analysis = analyze_food_habits(df.copy())

    # 4. Contexte RAG
    context_docs = search_user_data(user_id, question, top_k=5)
    context = "\n".join([d['text'] for d in context_docs])

    # 5. Formate les patterns
    patterns_str = format_patterns(
        day_analysis, cat_analysis,
        freq_analysis, trend_analysis, food_analysis
    )

    # 6. Appel LLM
    chain = behavioral_prompt | llm
    response = chain.invoke({
        "patterns": patterns_str,
        "context": context,
        "question": question
    })

    return {
        "agent": "behavioral_agent",
        "question": question,
        "answer": response.content,
        "behavior": {
            "day_analysis": day_analysis,
            "category_analysis": cat_analysis,
            "frequency_analysis": freq_analysis,
            "trend_analysis": trend_analysis,
            "food_analysis": food_analysis,
        }
    }