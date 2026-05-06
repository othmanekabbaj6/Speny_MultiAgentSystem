from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from data.firebase_client import get_user
from agents.budget_agent import run_budget_agent
from agents.goal_agent import run_goal_agent
from agents.anomaly_agent import run_anomaly_agent
from agents.behavioral_agent import run_behavioral_agent
from agents.persona_agent import run_persona_agent
from agents.forecast_agent import run_forecast_agent
from agents.simulation_agent import run_simulation_agent
from agents.retrieval_agent import run_retrieval_agent
from config.settings import settings
from datetime import datetime, timezone
import logging

logger = logging.getLogger(__name__)

llm = ChatGroq(
    api_key=settings.groq_api_key,
    model_name=settings.llm_model
)

advice_prompt = PromptTemplate(
    input_variables=[
        "user_name", "budget_summary", "goal_summary",
        "anomaly_summary", "behavioral_summary", "persona_summary",
        "forecast_summary", "question"
    ],
    template="""
Tu es SpendWise, un conseiller financier personnel IA bienveillant et expert.
Tu as analysé en profondeur les finances de {user_name} et tu dois lui donner
un conseil global, cohérent et actionnable. Réponds en français.

─── ANALYSE BUDGÉTAIRE ───────────────────────────────────────
{budget_summary}

─── OBJECTIFS FINANCIERS ────────────────────────────────────
{goal_summary}

─── ANOMALIES DÉTECTÉES ─────────────────────────────────────
{anomaly_summary}

─── HABITUDES DE DÉPENSES ───────────────────────────────────
{behavioral_summary}

─── PROFIL FINANCIER ────────────────────────────────────────
{persona_summary}

─── PRÉVISIONS DU MOIS PROCHAIN ─────────────────────────────
{forecast_summary}

─── QUESTION DE L'UTILISATEUR ───────────────────────────────
{question}

Ta réponse doit :
1. Commencer par une synthèse de la situation financière globale (2-3 phrases)
2. Répondre directement à la question posée
3. Donner 3 recommandations prioritaires numérotées, concrètes et personnalisées
4. Terminer par un message d'encouragement court

Sois précis avec les chiffres. Évite les généralités.
Adapte le ton au profil de l'utilisateur.
"""
)

# ─── Extraction des résumés ────────────────────────────────────
def extract_budget_summary(result: dict) -> str:
    try:
        budgets = result.get('budgets', [])
        spending = result.get('spending_by_category', {})
        spending_no_budget = result.get('spending_no_budget', {})

        lines = []
        total = sum(spending.values()) + sum(spending_no_budget.values())
        lines.append(f"Dépenses totales du mois : {round(total, 2)} MAD")

        over_budget = []
        for b in budgets:
            cat = b.get('category', '')
            limit = b.get('limit_amount', 0)
            # Recherche insensible à la casse
            spent = next(
                (v for k, v in spending.items() if k.lower() == cat.lower()), 0
            )
            pct = round(spent / limit * 100, 1) if limit > 0 else 0

            if pct > 100:
                over_budget.append(
                    f"⚠️ {cat} dépassé : {round(spent, 2)} MAD "
                    f"/ {limit} MAD ({pct}%)"
                )
            else:
                lines.append(
                    f"- {cat} : {round(spent, 2)} MAD "
                    f"/ {limit} MAD ({pct}%)"
                )

        if over_budget:
            lines.extend(over_budget)
        else:
            lines.append("✅ Aucun budget dépassé ce mois-ci")

        if spending_no_budget:
            top = sorted(spending_no_budget.items(), key=lambda x: -x[1])[:3]
            lines.append("Sans budget défini :")
            for cat, amt in top:
                lines.append(f"  - {cat} : {round(amt, 2)} MAD")

        return "\n".join(lines)
    except Exception as e:
        logger.error(f"extract_budget_summary error: {e}")
        return result.get('answer', 'Non disponible')[:300]


def extract_goal_summary(result: dict) -> str:
    try:
        goals = result.get('goals', [])
        summary = result.get('monthly_summary', {})

        income = summary.get('income', 0)
        expenses = summary.get('expenses', 0)
        savings = summary.get('savings', 0)

        lines = [
            f"Revenus du mois : {round(income, 2)} MAD",
            f"Dépenses du mois : {round(expenses, 2)} MAD",
            f"Épargne mensuelle : {round(savings, 2)} MAD",
        ]

        for g in goals[:4]:
            progress = g.get('progress_pct', 0)
            months = g.get('months_to_complete')
            months_str = f"{months} mois" if months and months != float('inf') else "∞"
            lines.append(
                f"- {g.get('title')} : {progress:.1f}% "
                f"(reste ~{months_str})"
            )

        return "\n".join(lines)
    except Exception as e:
        logger.error(f"extract_goal_summary error: {e}")
        return result.get('answer', 'Non disponible')[:300]


def extract_anomaly_summary(result: dict) -> str:
    try:
        count = result.get('anomalies_count', 0)
        anomalies = result.get('anomalies', [])

        if not anomalies:
            monthly = result.get('monthly_results', {})
            for month_data in monthly.values():
                anomalies.extend(month_data.get('anomalies', []))
            count = len(anomalies)

        if count == 0:
            return "✅ Aucune anomalie détectée"

        lines = [f"⚠️ {count} anomalie(s) détectée(s) :"]
        for a in anomalies[:4]:
            lines.append(
                f"  - {a.get('merchant_name', '?')} : "
                f"{a.get('amount', 0)} MAD "
                f"({a.get('category', '?')})"
            )
        return "\n".join(lines)
    except Exception as e:
        logger.error(f"extract_anomaly_summary error: {e}")
        return result.get('answer', 'Non disponible')[:300]


def extract_behavioral_summary(result: dict) -> str:
    try:
        b = result.get('behavior', {})
        day = b.get('day_analysis', {})
        cat = b.get('category_analysis', {})
        trend = b.get('trend_analysis', {})
        food = b.get('food_analysis', {})

        lines = [
            f"Jour le plus dépensier : {day.get('top_day', '?')} "
            f"({day.get('top_day_total', 0)} MAD)",
            f"Catégorie dominante : {cat.get('top_category', '?')} "
            f"({cat.get('top_category_pct', 0)}%)",
            f"Tendance : {trend.get('trend_direction', '?')} "
            f"({trend.get('trend_pct', 0)}%)",
        ]
        if food.get('found'):
            lines.append(
                f"Food : {food.get('food_pct_of_total', 0)}% des dépenses, "
                f"{food.get('avg_food_transactions_per_week', 0)} fois/semaine"
            )
        return "\n".join(lines)
    except Exception as e:
        logger.error(f"extract_behavioral_summary error: {e}")
        return result.get('answer', 'Non disponible')[:300]


def extract_persona_summary(result: dict) -> str:
    try:
        persona = result.get('persona', {})
        scores = result.get('scores', {})
        lines = [
            f"Type : {persona.get('persona', '?')}",
            f"Description : {persona.get('description', '')}",
            f"Score global : {scores.get('global', 0)}/100",
            f"Épargne : {scores.get('savings', 0)}/100 "
            f"(taux {result.get('savings_ratio', 0)}%)",
            f"Discipline budget : {scores.get('budget_discipline', 0)}/100",
            f"Régularité : {scores.get('regularity', 0)}/100",
            f"Tolérance au risque : {result.get('risk_tolerance', '?')}",
        ]
        return "\n".join(lines)
    except Exception as e:
        logger.error(f"extract_persona_summary error: {e}")
        return result.get('answer', 'Non disponible')[:300]


def extract_forecast_summary(result: dict) -> str:
    try:
        total = result.get('total_predicted', 0)
        at_risk = result.get('at_risk_categories', [])
        forecasts = result.get('forecasts', [])

        lines = [f"Total prévu le mois prochain : {round(total, 2)} MAD"]

        if at_risk:
            lines.append("⚠️ Catégories à risque :")
            for f in at_risk:
                lines.append(
                    f"  - {f['category']} : {f['predicted']} MAD prédit"
                )

        for f in forecasts[:4]:
            trend = f"↑+{f['trend_pct']}%" if f['trend_pct'] > 0 else \
                    f"↓{f['trend_pct']}%" if f['trend_pct'] < 0 else "→stable"
            lines.append(
                f"- {f['category']} : {f['predicted']} MAD ({trend})"
            )

        return "\n".join(lines)
    except Exception as e:
        logger.error(f"extract_forecast_summary error: {e}")
        return result.get('answer', 'Non disponible')[:300]


# ─── Modes d'exécution ─────────────────────────────────────────
AGENT_MODES = {
    "full":     ["budget", "goal", "anomaly", "behavioral", "persona", "forecast"],
    "quick":    ["budget", "goal", "anomaly"],
    "forecast": ["forecast", "persona"],
    "profile":  ["behavioral", "persona"],
}

# ─── Agent principal ───────────────────────────────────────────
def run_advice_agent(
    user_id: str,
    question: str = None,
    mode: str = "full",
    month: int = None,
    year: int = None
) -> dict:
    """
    Orchestrateur central — agrège tous les agents et génère
    une recommandation financière globale.

    mode :
      - "full"     : tous les agents (complet, ~60s)
      - "quick"    : budget + goal + anomaly (~20s)
      - "forecast" : forecast + persona (~30s)
      - "profile"  : behavioral + persona (~15s)
    """
    if question is None:
        question = "Donne-moi une analyse complète de ma situation financière et tes recommandations."

    now = datetime.now(timezone.utc)

    # Par défaut : mois précédent (plus de données disponibles)
    if month is None or year is None:
        if now.month == 1:
            month = 12
            year = now.year - 1
        else:
            month = now.month - 1
            year = now.year

    # 1. Infos utilisateur
    user = get_user(user_id) or {}
    user_name = user.get('display_name', 'utilisateur')

    agents_to_run = AGENT_MODES.get(mode, AGENT_MODES["full"])
    logger.info(f"Advice Agent — mode: {mode}, agents: {agents_to_run}")

    # 2. Exécute les agents sélectionnés
    results = {}

    if "budget" in agents_to_run:
        logger.info("Running Budget Agent...")
        try:
            results['budget'] = run_budget_agent(user_id, month=month, year=year)
        except Exception as e:
            logger.error(f"Budget agent error: {e}")
            results['budget'] = {"answer": "Non disponible", "budgets": [],
                                 "spending_by_category": {}, "spending_no_budget": {}}

    if "goal" in agents_to_run:
        logger.info("Running Goal Agent...")
        try:
            results['goal'] = run_goal_agent(user_id, month=month, year=year)
        except Exception as e:
            logger.error(f"Goal agent error: {e}")
            results['goal'] = {"answer": "Non disponible", "goals": [], "monthly_summary": {}}

    if "anomaly" in agents_to_run:
        logger.info("Running Anomaly Agent...")
        try:
            results['anomaly'] = run_anomaly_agent(
                user_id, month=month, year=year, mode="monthly"
            )
        except Exception as e:
            logger.error(f"Anomaly agent error: {e}")
            results['anomaly'] = {"answer": "Non disponible", "anomalies": [],
                                  "anomalies_count": 0, "monthly_results": {}}

    if "behavioral" in agents_to_run:
        logger.info("Running Behavioral Agent...")
        try:
            results['behavioral'] = run_behavioral_agent(user_id)
        except Exception as e:
            logger.error(f"Behavioral agent error: {e}")
            results['behavioral'] = {"answer": "Non disponible", "behavior": {}}

    if "persona" in agents_to_run:
        logger.info("Running Persona Agent...")
        try:
            results['persona'] = run_persona_agent(user_id)
        except Exception as e:
            logger.error(f"Persona agent error: {e}")
            results['persona'] = {"answer": "Non disponible", "persona": {},
                                  "scores": {}, "savings_ratio": 0, "risk_tolerance": "?"}

    if "forecast" in agents_to_run:
        logger.info("Running Forecast Agent...")
        try:
            results['forecast'] = run_forecast_agent(user_id)
        except Exception as e:
            logger.error(f"Forecast agent error: {e}")
            results['forecast'] = {"answer": "Non disponible", "forecasts": [],
                                   "total_predicted": 0, "at_risk_categories": []}

    # 3. Extrait les résumés
    budget_summary     = extract_budget_summary(results.get('budget', {}))
    goal_summary       = extract_goal_summary(results.get('goal', {}))
    anomaly_summary    = extract_anomaly_summary(results.get('anomaly', {}))
    behavioral_summary = extract_behavioral_summary(results.get('behavioral', {}))
    persona_summary    = extract_persona_summary(results.get('persona', {}))
    forecast_summary   = extract_forecast_summary(results.get('forecast', {}))

    # 4. Appel LLM final
    chain = advice_prompt | llm
    response = chain.invoke({
        "user_name": user_name,
        "budget_summary": budget_summary,
        "goal_summary": goal_summary,
        "anomaly_summary": anomaly_summary,
        "behavioral_summary": behavioral_summary,
        "persona_summary": persona_summary,
        "forecast_summary": forecast_summary,
        "question": question
    })

    return {
        "agent": "advice_agent",
        "mode": mode,
        "question": question,
        "answer": response.content,
        "agent_results": results,
        "summaries": {
            "budget": budget_summary,
            "goal": goal_summary,
            "anomaly": anomaly_summary,
            "behavioral": behavioral_summary,
            "persona": persona_summary,
            "forecast": forecast_summary,
        }
    }