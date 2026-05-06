from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from config.settings import settings
import logging

logger = logging.getLogger(__name__)

llm = ChatGroq(
    api_key=settings.groq_api_key,
    model_name=settings.llm_model
)

explanation_prompt = PromptTemplate(
    input_variables=["agent_name", "technical_output", "audience", "question"],
    template="""
Tu es SpendWise, un assistant financier qui traduit les analyses techniques
en langage simple et compréhensible. Réponds en français.

AGENT SOURCE : {agent_name}
AUDIENCE CIBLE : {audience}

OUTPUT TECHNIQUE À TRADUIRE :
{technical_output}

QUESTION ORIGINALE : {question}

Traduis cet output en langage naturel simple :
- Utilise des phrases courtes et claires
- Évite le jargon technique
- Ajoute des emojis pour rendre la lecture agréable
- Structure avec des titres simples si nécessaire
- Pour les chiffres, arrondis et contextualise (ex: "c'est l'équivalent de 3 repas")
- Termine par 1 conseil pratique immédiat

Public {audience} : adapte le niveau de détail en conséquence.
"""
)

simplify_prompt = PromptTemplate(
    input_variables=["concept", "context"],
    template="""
Explique ce concept financier en français simple, comme si tu parlais
à quelqu'un qui ne connaît pas la finance.

CONCEPT : {concept}
CONTEXTE : {context}

En 2-3 phrases maximum, sans jargon. Utilise une analogie du quotidien si possible.
"""
)

# ─── Traduction d'un output d'agent ───────────────────────────
def explain_agent_output(
    agent_name: str,
    technical_output: dict,
    question: str = None,
    audience: str = "grand public"
) -> dict:
    """
    Traduit l'output technique d'un agent en langage naturel simple.

    audience : "grand public" | "expert" | "débutant"
    """
    if question is None:
        question = "Explique-moi mes finances simplement."

    # Extrait le contenu pertinent selon l'agent
    content = _extract_content_for_agent(agent_name, technical_output)

    chain = explanation_prompt | llm
    response = chain.invoke({
        "agent_name": agent_name,
        "technical_output": content,
        "audience": audience,
        "question": question
    })

    return {
        "agent": "explanation_agent",
        "source_agent": agent_name,
        "audience": audience,
        "explanation": response.content,
        "original_answer": technical_output.get('answer', '')
    }

# ─── Extraction du contenu pertinent par agent ────────────────
def _extract_content_for_agent(agent_name: str, result: dict) -> str:
    """
    Extrait les données clés d'un résultat d'agent
    pour les passer à l'Explanation Agent.
    """
    lines = []

    if agent_name == "budget_agent":
        spending = result.get('spending_by_category', {})
        budgets = result.get('budgets', [])
        lines.append(f"Dépenses par catégorie ce mois : {spending}")
        for b in budgets:
            cat = b.get('category', '')
            limit = b.get('limit_amount', 0)
            spent = next(
                (v for k, v in spending.items() if k.lower() == cat.lower()), 0
            )
            pct = round(spent / limit * 100, 1) if limit > 0 else 0
            lines.append(f"Budget {cat} : {spent} MAD / {limit} MAD ({pct}%)")

    elif agent_name == "goal_agent":
        summary = result.get('monthly_summary', {})
        goals = result.get('goals', [])
        lines.append(f"Revenus : {summary.get('income', 0)} MAD")
        lines.append(f"Dépenses : {summary.get('expenses', 0)} MAD")
        lines.append(f"Épargne : {summary.get('savings', 0)} MAD")
        for g in goals:
            lines.append(
                f"Objectif {g.get('title')} : "
                f"{g.get('progress_pct', 0):.1f}% atteint, "
                f"reste {g.get('months_to_complete', '?')} mois"
            )

    elif agent_name == "anomaly_agent":
        count = result.get('anomalies_count', 0)
        anomalies = result.get('anomalies', [])
        lines.append(f"Nombre d'anomalies détectées : {count}")
        for a in anomalies[:5]:
            lines.append(
                f"- {a.get('merchant_name', '?')} : "
                f"{a.get('amount', 0)} MAD "
                f"(déviation {a.get('deviation_from_mean', 0)}x la normale)"
            )

    elif agent_name == "behavioral_agent":
        b = result.get('behavior', {})
        day = b.get('day_analysis', {})
        cat = b.get('category_analysis', {})
        trend = b.get('trend_analysis', {})
        food = b.get('food_analysis', {})
        lines.append(f"Jour le plus dépensier : {day.get('top_day')} ({day.get('top_day_total')} MAD)")
        lines.append(f"Catégorie principale : {cat.get('top_category')} ({cat.get('top_category_pct')}%)")
        lines.append(f"Tendance : {trend.get('trend_direction')} de {trend.get('trend_pct')}%")
        if food.get('found'):
            lines.append(
                f"Food : {food.get('food_pct_of_total')}% du budget, "
                f"{food.get('avg_food_transactions_per_week')} fois/semaine, "
                f"{food.get('food_avg_per_transaction')} MAD en moyenne"
            )

    elif agent_name == "persona_agent":
        persona = result.get('persona', {})
        scores = result.get('scores', {})
        lines.append(f"Profil : {persona.get('persona')}")
        lines.append(f"Score global : {scores.get('global')}/100")
        lines.append(f"Taux d'épargne : {result.get('savings_ratio')}%")
        lines.append(f"Discipline budget : {scores.get('budget_discipline')}/100")
        lines.append(f"Régularité : {scores.get('regularity')}/100")
        lines.append(f"Tolérance au risque : {result.get('risk_tolerance')}")

    elif agent_name == "forecast_agent":
        total = result.get('total_predicted', 0)
        forecasts = result.get('forecasts', [])
        at_risk = result.get('at_risk_categories', [])
        lines.append(f"Total prévu le mois prochain : {total} MAD")
        for f in forecasts[:5]:
            lines.append(
                f"- {f['category']} : {f['predicted']} MAD "
                f"(tendance {f['trend_pct']}%)"
            )
        if at_risk:
            lines.append(f"Catégories à risque de dépassement : "
                        f"{[f['category'] for f in at_risk]}")

    elif agent_name == "simulation_agent":
        r = result.get('result', {})
        baseline = result.get('baseline', {})
        lines.append(f"Type de simulation : {r.get('type')}")
        lines.append(f"Épargne actuelle : {baseline.get('monthly_savings')} MAD/mois")
        if r.get('type') == 'category_reduction':
            lines.append(f"Réduction {r.get('category')} de {r.get('reduction_pct')}%")
            lines.append(f"Économie mensuelle : {r.get('reduction_amount')} MAD")
            lines.append(f"Total économisé sur {r.get('simulation_months')} mois : {r.get('total_saved_over_period')} MAD")
        for g in r.get('goal_impacts', []):
            lines.append(
                f"Impact sur {g['title']} : "
                f"{g['months_before']} → {g['months_after']} mois "
                f"(gain {g['time_saved_months']} mois)"
            )

    elif agent_name == "advice_agent":
        summaries = result.get('summaries', {})
        for k, v in summaries.items():
            if v:
                lines.append(f"[{k.upper()}] {v[:200]}")

    else:
        lines.append(result.get('answer', str(result))[:500])

    return "\n".join(lines)

# ─── Explication d'un concept financier ───────────────────────
def explain_concept(concept: str, context: str = "") -> str:
    """
    Explique un concept financier en langage simple.
    Ex: explain_concept("taux d'épargne", "j'ai un salaire de 10000 MAD")
    """
    chain = simplify_prompt | llm
    response = chain.invoke({
        "concept": concept,
        "context": context
    })
    return response.content

# ─── Agent principal ───────────────────────────────────────────
def run_explanation_agent(
    agent_name: str,
    agent_result: dict,
    question: str = None,
    audience: str = "grand public"
) -> dict:
    """
    Traduit l'output de n'importe quel agent en langage naturel simple.

    Paramètres :
    - agent_name  : nom de l'agent source (ex: "budget_agent")
    - agent_result: résultat brut retourné par l'agent
    - question    : question originale de l'utilisateur
    - audience    : "grand public" | "expert" | "débutant"
    """
    return explain_agent_output(agent_name, agent_result, question, audience)