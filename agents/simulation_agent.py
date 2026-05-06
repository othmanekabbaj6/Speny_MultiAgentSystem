import pandas as pd
import numpy as np
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from data.firebase_client import get_transactions, get_budgets, get_goals
from rag.query_engine import search_user_data
from config.settings import settings
from datetime import datetime, timezone
import logging

logger = logging.getLogger(__name__)

llm = ChatGroq(
    api_key=settings.groq_api_key,
    model_name=settings.llm_model
)

simulation_prompt = PromptTemplate(
    input_variables=["scenario", "results", "context", "question"],
    template="""
Tu es un conseiller financier expert en simulation et planification.
Analyse les résultats de simulation suivants et réponds en français.

SCÉNARIO SIMULÉ :
{scenario}

RÉSULTATS DE LA SIMULATION :
{results}

CONTEXTE SUPPLÉMENTAIRE :
{context}

QUESTION : {question}

Sur la base de cette simulation :
1. Explique clairement l'impact financier du scénario
2. Indique si le scénario est réaliste et atteignable
3. Montre l'impact sur les objectifs financiers
4. Propose des étapes concrètes pour mettre en œuvre ce scénario
5. Mentionne les risques ou effets secondaires potentiels

Sois précis avec les chiffres et actionnable dans tes recommandations.
"""
)

# ─── Calcul des stats de base ─────────────────────────────────
def get_financial_baseline(
    transactions: list[dict],
    months: int = 3
) -> dict:
    """
    Calcule la situation financière actuelle
    sur les derniers mois comme référence.
    """
    now = datetime.now(timezone.utc)
    recent = []
    for t in transactions:
        date = t.get('date')
        if hasattr(date, 'tzinfo') and date.tzinfo is None:
            date = date.replace(tzinfo=timezone.utc)
        if date and (now - date).days <= months * 30:
            recent.append(t)

    income = sum(
        t.get('amount', 0) for t in recent
        if t.get('type') == 'income'
    )
    expenses = sum(
        t.get('amount', 0) for t in recent
        if t.get('type') == 'expense'
    )

    # Dépenses par catégorie
    by_category = {}
    for t in recent:
        if t.get('type') == 'expense':
            cat = t.get('category', 'Other')
            by_category[cat] = by_category.get(cat, 0) + t.get('amount', 0)

    # Moyenne mensuelle
    monthly_income = round(income / months, 2)
    monthly_expenses = round(expenses / months, 2)
    monthly_savings = round(monthly_income - monthly_expenses, 2)

    monthly_by_category = {
        cat: round(amt / months, 2)
        for cat, amt in by_category.items()
    }

    return {
        'monthly_income': monthly_income,
        'monthly_expenses': monthly_expenses,
        'monthly_savings': monthly_savings,
        'savings_rate': round(monthly_savings / monthly_income * 100, 1) if monthly_income > 0 else 0,
        'by_category': monthly_by_category,
        'months_analyzed': months
    }

# ─── Simulation : Réduction d'une catégorie ──────────────────
def simulate_category_reduction(
    baseline: dict,
    goals: list[dict],
    category: str,
    reduction_pct: float,
    months: int = 6
) -> dict:
    """
    Simule l'impact d'une réduction de dépenses
    dans une catégorie sur N mois.
    """
    current_spending = baseline['by_category'].get(category, 0)
    reduction_amount = round(current_spending * reduction_pct / 100, 2)
    new_spending = round(current_spending - reduction_amount, 2)
    new_monthly_savings = round(baseline['monthly_savings'] + reduction_amount, 2)
    total_saved = round(reduction_amount * months, 2)

    # Impact sur les goals
    goal_impacts = []
    for g in goals:
        if g.get('status') != 'active':
            continue
        remaining = g.get('target_amount', 0) - g.get('current_amount', 0)
        if remaining <= 0:
            continue

        # Mois nécessaires AVANT simulation
        months_before = round(
            remaining / baseline['monthly_savings'], 1
        ) if baseline['monthly_savings'] > 0 else float('inf')

        # Mois nécessaires APRÈS simulation
        months_after = round(
            remaining / new_monthly_savings, 1
        ) if new_monthly_savings > 0 else float('inf')

        time_saved = round(months_before - months_after, 1) if \
            months_before != float('inf') and months_after != float('inf') else 0

        goal_impacts.append({
            'title': g.get('title'),
            'remaining': round(remaining, 2),
            'months_before': months_before,
            'months_after': months_after,
            'time_saved_months': time_saved
        })

    return {
        'type': 'category_reduction',
        'category': category,
        'reduction_pct': reduction_pct,
        'current_spending': current_spending,
        'new_spending': new_spending,
        'reduction_amount': reduction_amount,
        'new_monthly_savings': new_monthly_savings,
        'total_saved_over_period': total_saved,
        'simulation_months': months,
        'goal_impacts': goal_impacts
    }

# ─── Simulation : Augmentation de revenus ────────────────────
def simulate_income_increase(
    baseline: dict,
    goals: list[dict],
    increase_pct: float,
    months: int = 6
) -> dict:
    """
    Simule l'impact d'une augmentation de revenus sur N mois.
    """
    increase_amount = round(baseline['monthly_income'] * increase_pct / 100, 2)
    new_income = round(baseline['monthly_income'] + increase_amount, 2)
    new_savings = round(baseline['monthly_savings'] + increase_amount, 2)
    new_savings_rate = round(new_savings / new_income * 100, 1)
    total_extra = round(increase_amount * months, 2)

    goal_impacts = []
    for g in goals:
        if g.get('status') != 'active':
            continue
        remaining = g.get('target_amount', 0) - g.get('current_amount', 0)
        if remaining <= 0:
            continue

        months_before = round(
            remaining / baseline['monthly_savings'], 1
        ) if baseline['monthly_savings'] > 0 else float('inf')

        months_after = round(
            remaining / new_savings, 1
        ) if new_savings > 0 else float('inf')

        time_saved = round(months_before - months_after, 1) if \
            months_before != float('inf') and months_after != float('inf') else 0

        goal_impacts.append({
            'title': g.get('title'),
            'remaining': round(remaining, 2),
            'months_before': months_before,
            'months_after': months_after,
            'time_saved_months': time_saved
        })

    return {
        'type': 'income_increase',
        'increase_pct': increase_pct,
        'current_income': baseline['monthly_income'],
        'new_income': new_income,
        'increase_amount': increase_amount,
        'new_monthly_savings': new_savings,
        'new_savings_rate': new_savings_rate,
        'total_extra_over_period': total_extra,
        'simulation_months': months,
        'goal_impacts': goal_impacts
    }

# ─── Simulation : Objectif spécifique ─────────────────────────
def simulate_goal_achievement(
    baseline: dict,
    goal: dict,
    extra_monthly_savings: float = 0
) -> dict:
    """
    Simule combien de temps pour atteindre un objectif
    avec ou sans épargne supplémentaire.
    """
    remaining = goal.get('target_amount', 0) - goal.get('current_amount', 0)
    if remaining <= 0:
        return {
            'type': 'goal_achievement',
            'goal_title': goal.get('title'),
            'status': 'already_achieved',
            'months_needed': 0
        }

    monthly_savings = baseline['monthly_savings']
    monthly_with_extra = monthly_savings + extra_monthly_savings

    months_without_extra = round(
        remaining / monthly_savings, 1
    ) if monthly_savings > 0 else float('inf')

    months_with_extra = round(
        remaining / monthly_with_extra, 1
    ) if monthly_with_extra > 0 else float('inf')

    time_saved = round(months_without_extra - months_with_extra, 1) if \
        months_without_extra != float('inf') and \
        months_with_extra != float('inf') else 0

    return {
        'type': 'goal_achievement',
        'goal_title': goal.get('title'),
        'target': goal.get('target_amount', 0),
        'current': goal.get('current_amount', 0),
        'remaining': round(remaining, 2),
        'monthly_savings_current': monthly_savings,
        'monthly_savings_with_extra': monthly_with_extra,
        'extra_monthly': extra_monthly_savings,
        'months_without_extra': months_without_extra,
        'months_with_extra': months_with_extra,
        'time_saved_months': time_saved
    }

# ─── Formatage des résultats ──────────────────────────────────
def format_simulation_results(result: dict, baseline: dict) -> str:
    lines = []

    lines.append("📊 SITUATION ACTUELLE :")
    lines.append(f"  - Revenus mensuels : {baseline['monthly_income']} MAD")
    lines.append(f"  - Dépenses mensuelles : {baseline['monthly_expenses']} MAD")
    lines.append(f"  - Épargne mensuelle : {baseline['monthly_savings']} MAD")
    lines.append(f"  - Taux d'épargne : {baseline['savings_rate']}%")

    if result['type'] == 'category_reduction':
        lines.append(f"\n✂️ SIMULATION — Réduction {result['category']} de {result['reduction_pct']}% :")
        lines.append(f"  - Dépenses actuelles {result['category']} : {result['current_spending']} MAD/mois")
        lines.append(f"  - Nouvelles dépenses {result['category']} : {result['new_spending']} MAD/mois")
        lines.append(f"  - Économie mensuelle : {result['reduction_amount']} MAD")
        lines.append(f"  - Nouvelle épargne mensuelle : {result['new_monthly_savings']} MAD")
        lines.append(
            f"  - Total économisé sur {result['simulation_months']} mois : "
            f"{result['total_saved_over_period']} MAD"
        )

    elif result['type'] == 'income_increase':
        lines.append(f"\n💰 SIMULATION — Augmentation revenus de {result['increase_pct']}% :")
        lines.append(f"  - Revenus actuels : {result['current_income']} MAD/mois")
        lines.append(f"  - Nouveaux revenus : {result['new_income']} MAD/mois")
        lines.append(f"  - Gain mensuel : {result['increase_amount']} MAD")
        lines.append(f"  - Nouvelle épargne : {result['new_monthly_savings']} MAD/mois")
        lines.append(f"  - Nouveau taux d'épargne : {result['new_savings_rate']}%")
        lines.append(
            f"  - Gain total sur {result['simulation_months']} mois : "
            f"{result['total_extra_over_period']} MAD"
        )

    elif result['type'] == 'goal_achievement':
        if result.get('status') == 'already_achieved':
            lines.append(f"\n✅ Objectif '{result['goal_title']}' déjà atteint !")
        else:
            lines.append(f"\n🎯 SIMULATION — Objectif '{result['goal_title']}' :")
            lines.append(f"  - Montant restant : {result['remaining']} MAD")
            lines.append(
                f"  - Sans épargne supplémentaire : "
                f"{result['months_without_extra']} mois"
            )
            if result['extra_monthly'] > 0:
                lines.append(
                    f"  - Avec +{result['extra_monthly']} MAD/mois : "
                    f"{result['months_with_extra']} mois"
                )
                lines.append(
                    f"  - Temps gagné : {result['time_saved_months']} mois"
                )

    # Impact sur les goals
    if result.get('goal_impacts'):
        lines.append("\n🎯 IMPACT SUR LES OBJECTIFS :")
        for g in result['goal_impacts']:
            if g['months_before'] == float('inf'):
                before_str = "∞"
            else:
                before_str = f"{g['months_before']} mois"

            if g['months_after'] == float('inf'):
                after_str = "∞"
            else:
                after_str = f"{g['months_after']} mois"

            lines.append(
                f"  - {g['title']} : {before_str} → {after_str} "
                f"(gain: {g['time_saved_months']} mois)"
            )

    return "\n".join(lines)

def format_scenario(result: dict) -> str:
    if result['type'] == 'category_reduction':
        return (
            f"Réduction des dépenses '{result['category']}' "
            f"de {result['reduction_pct']}% sur "
            f"{result['simulation_months']} mois"
        )
    elif result['type'] == 'income_increase':
        return (
            f"Augmentation des revenus de {result['increase_pct']}% "
            f"sur {result['simulation_months']} mois"
        )
    elif result['type'] == 'goal_achievement':
        return f"Atteindre l'objectif '{result['goal_title']}'"
    return "Simulation financière"

# ─── Agent principal ───────────────────────────────────────────
def run_simulation_agent(
    user_id: str,
    scenario: str = "category_reduction",
    category: str = "Food",
    reduction_pct: float = 20.0,
    income_increase_pct: float = 10.0,
    goal_index: int = 0,
    extra_monthly_savings: float = 0,
    months: int = 6,
    question: str = None
) -> dict:
    """
    Lance une simulation financière.

    Scénarios disponibles :
    - "category_reduction" : réduire une catégorie de X%
    - "income_increase"    : augmenter les revenus de X%
    - "goal_achievement"   : combien de temps pour atteindre un goal
    """
    if question is None:
        if scenario == "category_reduction":
            question = f"Quel serait l'impact si je réduisais mes dépenses {category} de {reduction_pct}% ?"
        elif scenario == "income_increase":
            question = f"Quel serait l'impact si mes revenus augmentaient de {income_increase_pct}% ?"
        else:
            question = "Quand vais-je atteindre mon objectif financier ?"

    # 1. Données
    transactions = get_transactions(user_id, limit=500)
    goals = get_goals(user_id)

    # 2. Baseline financière
    baseline = get_financial_baseline(transactions, months=3)

    # 3. Lance le scénario
    if scenario == "category_reduction":
        result = simulate_category_reduction(
            baseline, goals, category, reduction_pct, months
        )
    elif scenario == "income_increase":
        result = simulate_income_increase(
            baseline, goals, income_increase_pct, months
        )
    elif scenario == "goal_achievement":
        if not goals or goal_index >= len(goals):
            return {"error": "Objectif non trouvé"}
        result = simulate_goal_achievement(
            baseline, goals[goal_index], extra_monthly_savings
        )
    else:
        return {"error": f"Scénario inconnu : {scenario}"}

    # 4. Contexte RAG
    context_docs = search_user_data(user_id, question, top_k=5)
    context = "\n".join([d['text'] for d in context_docs])

    # 5. Formate
    scenario_str = format_scenario(result)
    results_str = format_simulation_results(result, baseline)

    # 6. Appel LLM
    chain = simulation_prompt | llm
    response = chain.invoke({
        "scenario": scenario_str,
        "results": results_str,
        "context": context,
        "question": question
    })

    return {
        "agent": "simulation_agent",
        "scenario": scenario,
        "question": question,
        "answer": response.content,
        "baseline": baseline,
        "result": result
    }