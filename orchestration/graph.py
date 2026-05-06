"""
orchestration/graph.py
─────────────────────
Workflow conditionnel LangGraph pour SpendWise.

Au lieu d'appeler tous les agents en aveugle, le graph :
1. Analyse la question de l'utilisateur (RouterNode)
2. Décide quels agents appeler (edges conditionnels)
3. Exécute uniquement les agents pertinents
4. Agrège les résultats et génère la réponse finale

Flux :
    START
      ↓
    router          ← analyse la question et décide le mode
      ↓
  (conditionnel)
    ↓         ↓         ↓          ↓
  budget    forecast  anomaly   full_analysis
    ↓         ↓         ↓          ↓
    └─────────┴─────────┴──────────┘
                   ↓
              aggregator      ← fusionne les résultats
                   ↓
                 END
"""

from typing import TypedDict, Optional, List, Annotated
from langgraph.graph import StateGraph, END
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from config.settings import settings
from datetime import datetime, timezone
import logging
import operator

logger = logging.getLogger(__name__)

llm = ChatGroq(
    api_key=settings.groq_api_key,
    model_name=settings.llm_model
)

# ─── État partagé entre tous les nodes ────────────────────────

class SpendWiseState(TypedDict):
    # Entrée
    user_id:    str
    question:   str
    month:      Optional[int]
    year:       Optional[int]

    # Décision du router
    route:      Optional[str]   # budget | forecast | anomaly | behavioral | full

    # Résultats des agents
    budget_result:      Optional[dict]
    goal_result:        Optional[dict]
    anomaly_result:     Optional[dict]
    behavioral_result:  Optional[dict]
    persona_result:     Optional[dict]
    forecast_result:    Optional[dict]

    # Réponse finale
    final_answer:   Optional[str]
    agents_used:    Optional[List[str]]


# ─── Node 1 : Router ──────────────────────────────────────────

def router_node(state: SpendWiseState) -> SpendWiseState:
    """
    Analyse la question et décide quels agents appeler.
    Retourne une route parmi : budget | forecast | anomaly | behavioral | full
    """
    question = state["question"].lower()

    router_prompt = PromptTemplate.from_template(
        """Tu es un routeur pour un système d'agents financiers.
Analyse la question et réponds avec UNE SEULE route parmi ces options :

- budget     : questions sur les dépenses, budgets, dépassements
- forecast   : questions sur les prévisions, le mois prochain
- anomaly    : questions sur les anomalies, dépenses inhabituelles
- behavioral : questions sur les habitudes, comportements de dépense
- persona    : questions sur le profil financier, score, type
- full       : analyse complète, tout savoir, situation générale

Réponds avec UN SEUL MOT (la route), rien d'autre.

Question: {question}
Route:"""
    )

    try:
        response = (router_prompt | llm).invoke({"question": state["question"]})
        route = response.content.strip().lower().split()[0]

        valid_routes = {"budget", "forecast", "anomaly", "behavioral", "persona", "full"}
        if route not in valid_routes:
            route = "full"
    except Exception as e:
        logger.error(f"Router error: {e}")
        route = "full"

    logger.info(f"Router décision : '{state['question']}' → {route}")
    return {**state, "route": route, "agents_used": []}


# ─── Fonction de routage conditionnel ─────────────────────────

def route_decision(state: SpendWiseState) -> str:
    """Fonction appelée par LangGraph pour choisir le prochain node."""
    return state.get("route", "full")


# ─── Node 2a : Budget Node ────────────────────────────────────

def budget_node(state: SpendWiseState) -> SpendWiseState:
    """Appelle Budget Agent + Goal Agent."""
    from agents.budget_agent import run_budget_agent
    from agents.goal_agent import run_goal_agent

    month = state.get("month")
    year  = state.get("year")

    try:
        budget_result = run_budget_agent(state["user_id"], month=month, year=year)
    except Exception as e:
        logger.error(f"Budget node error: {e}")
        budget_result = {"answer": "Non disponible", "budgets": [], "spending_by_category": {}}

    try:
        goal_result = run_goal_agent(state["user_id"], month=month, year=year)
    except Exception as e:
        logger.error(f"Goal node error: {e}")
        goal_result = {"answer": "Non disponible", "goals": [], "monthly_summary": {}}

    return {
        **state,
        "budget_result":   budget_result,
        "goal_result":     goal_result,
        "agents_used":     ["budget_agent", "goal_agent"],
    }


# ─── Node 2b : Forecast Node ──────────────────────────────────

def forecast_node(state: SpendWiseState) -> SpendWiseState:
    """Appelle Forecast Agent + Persona Agent."""
    from agents.forecast_agent import run_forecast_agent
    from agents.persona_agent import run_persona_agent

    try:
        forecast_result = run_forecast_agent(state["user_id"])
    except Exception as e:
        logger.error(f"Forecast node error: {e}")
        forecast_result = {"answer": "Non disponible", "forecasts": [], "total_predicted": 0}

    try:
        persona_result = run_persona_agent(state["user_id"])
    except Exception as e:
        logger.error(f"Persona node error: {e}")
        persona_result = {"answer": "Non disponible", "persona": {}, "scores": {}}

    return {
        **state,
        "forecast_result": forecast_result,
        "persona_result":  persona_result,
        "agents_used":     ["forecast_agent", "persona_agent"],
    }


# ─── Node 2c : Anomaly Node ───────────────────────────────────

def anomaly_node(state: SpendWiseState) -> SpendWiseState:
    """Appelle Anomaly Agent."""
    from agents.anomaly_agent import run_anomaly_agent

    month = state.get("month")
    year  = state.get("year")

    try:
        anomaly_result = run_anomaly_agent(
            state["user_id"],
            mode="monthly" if month else "global",
            month=month,
            year=year
        )
    except Exception as e:
        logger.error(f"Anomaly node error: {e}")
        anomaly_result = {"answer": "Non disponible", "anomalies": [], "anomalies_count": 0}

    return {
        **state,
        "anomaly_result": anomaly_result,
        "agents_used":    ["anomaly_agent"],
    }


# ─── Node 2d : Behavioral Node ────────────────────────────────

def behavioral_node(state: SpendWiseState) -> SpendWiseState:
    """Appelle Behavioral Agent + Persona Agent."""
    from agents.behavioral_agent import run_behavioral_agent
    from agents.persona_agent import run_persona_agent

    try:
        behavioral_result = run_behavioral_agent(state["user_id"])
    except Exception as e:
        logger.error(f"Behavioral node error: {e}")
        behavioral_result = {"answer": "Non disponible", "behavior": {}}

    try:
        persona_result = run_persona_agent(state["user_id"])
    except Exception as e:
        logger.error(f"Persona node error: {e}")
        persona_result = {"answer": "Non disponible", "persona": {}, "scores": {}}

    return {
        **state,
        "behavioral_result": behavioral_result,
        "persona_result":    persona_result,
        "agents_used":       ["behavioral_agent", "persona_agent"],
    }


# ─── Node 2e : Persona Node ───────────────────────────────────

def persona_node(state: SpendWiseState) -> SpendWiseState:
    """Appelle Persona Agent."""
    from agents.persona_agent import run_persona_agent

    try:
        persona_result = run_persona_agent(state["user_id"])
    except Exception as e:
        logger.error(f"Persona node error: {e}")
        persona_result = {"answer": "Non disponible", "persona": {}, "scores": {}}

    return {
        **state,
        "persona_result": persona_result,
        "agents_used":    ["persona_agent"],
    }


# ─── Node 2f : Full Analysis Node ─────────────────────────────

def full_node(state: SpendWiseState) -> SpendWiseState:
    """Appelle tous les agents via l'Advice Agent."""
    from agents.advice_agent import run_advice_agent

    month = state.get("month")
    year  = state.get("year")

    try:
        advice_result = run_advice_agent(
            user_id=state["user_id"],
            question=state["question"],
            mode="full",
            month=month,
            year=year
        )
        # Récupère les résultats individuels depuis l'Advice Agent
        agent_results = advice_result.get("agent_results", {})
        return {
            **state,
            "budget_result":     agent_results.get("budget", {}),
            "goal_result":       agent_results.get("goal", {}),
            "anomaly_result":    agent_results.get("anomaly", {}),
            "behavioral_result": agent_results.get("behavioral", {}),
            "persona_result":    agent_results.get("persona", {}),
            "forecast_result":   agent_results.get("forecast", {}),
            "final_answer":      advice_result.get("answer", ""),
            "agents_used":       ["budget", "goal", "anomaly", "behavioral", "persona", "forecast"],
        }
    except Exception as e:
        logger.error(f"Full node error: {e}")
        return {**state, "final_answer": f"Erreur analyse complète: {e}", "agents_used": []}


# ─── Node 3 : Aggregator ──────────────────────────────────────

def aggregator_node(state: SpendWiseState) -> SpendWiseState:
    """
    Agrège les résultats des agents appelés et génère
    une réponse finale cohérente via le LLM.
    Ignoré si full_node a déjà généré final_answer.
    """
    # Si full_node a déjà répondu, on passe directement
    if state.get("final_answer"):
        return state

    # Collecte les réponses disponibles
    parts = []

    if state.get("budget_result"):
        parts.append(f"[BUDGET]\n{state['budget_result'].get('answer', '')[:400]}")

    if state.get("goal_result"):
        parts.append(f"[OBJECTIFS]\n{state['goal_result'].get('answer', '')[:400]}")

    if state.get("anomaly_result"):
        parts.append(f"[ANOMALIES]\n{state['anomaly_result'].get('answer', '')[:400]}")

    if state.get("behavioral_result"):
        parts.append(f"[HABITUDES]\n{state['behavioral_result'].get('answer', '')[:400]}")

    if state.get("persona_result"):
        parts.append(f"[PROFIL]\n{state['persona_result'].get('answer', '')[:400]}")

    if state.get("forecast_result"):
        parts.append(f"[PRÉVISIONS]\n{state['forecast_result'].get('answer', '')[:400]}")

    if not parts:
        return {**state, "final_answer": "Aucune donnée disponible."}

    aggregator_prompt = PromptTemplate.from_template(
        """Tu es SpendWise, un conseiller financier IA. Réponds en français.

Question de l'utilisateur : {question}

Données collectées par les agents :
{data}

Génère une réponse concise et directe qui :
1. Répond précisément à la question posée
2. S'appuie sur les chiffres fournis
3. Donne 1-2 recommandations pratiques maximum

Sois direct, utilise les vrais chiffres, max 5 phrases."""
    )

    try:
        response = (aggregator_prompt | llm).invoke({
            "question": state["question"],
            "data":     "\n\n".join(parts)
        })
        final_answer = response.content
    except Exception as e:
        logger.error(f"Aggregator error: {e}")
        final_answer = "\n\n".join(parts)

    return {**state, "final_answer": final_answer}


# ─── Construction du Graph ────────────────────────────────────

def build_graph():
    graph = StateGraph(SpendWiseState)

    # Nodes
    graph.add_node("router",     router_node)
    graph.add_node("budget",     budget_node)
    graph.add_node("forecast",   forecast_node)
    graph.add_node("anomaly",    anomaly_node)
    graph.add_node("behavioral", behavioral_node)
    graph.add_node("persona",    persona_node)
    graph.add_node("full",       full_node)
    graph.add_node("aggregator", aggregator_node)

    # Point d'entrée
    graph.set_entry_point("router")

    # Edges conditionnels depuis router
    graph.add_conditional_edges(
        "router",
        route_decision,
        {
            "budget":     "budget",
            "forecast":   "forecast",
            "anomaly":    "anomaly",
            "behavioral": "behavioral",
            "persona":    "persona",
            "full":       "full",
        }
    )

    # Tous les nodes mènent à l'aggregator
    for node in ["budget", "forecast", "anomaly", "behavioral", "persona"]:
        graph.add_edge(node, "aggregator")

    # full va directement à aggregator (qui détecte final_answer déjà rempli)
    graph.add_edge("full", "aggregator")

    # Fin
    graph.add_edge("aggregator", END)

    return graph.compile()


# ─── Instance globale du graph ─────────────────────────────────
spendwise_graph = build_graph()


# ─── Fonction principale d'appel ──────────────────────────────

def run_graph(
    user_id: str,
    question: str,
    month: int = None,
    year: int = None
) -> dict:
    """
    Point d'entrée principal du graph LangGraph.
    Analyse la question, route vers les bons agents, agrège la réponse.
    """
    now = datetime.now(timezone.utc)
    if month is None:
        month = now.month - 1 if now.month > 1 else 12
    if year is None:
        year = now.year if now.month > 1 else now.year - 1

    initial_state: SpendWiseState = {
        "user_id":           user_id,
        "question":          question,
        "month":             month,
        "year":              year,
        "route":             None,
        "budget_result":     None,
        "goal_result":       None,
        "anomaly_result":    None,
        "behavioral_result": None,
        "persona_result":    None,
        "forecast_result":   None,
        "final_answer":      None,
        "agents_used":       [],
    }

    result = spendwise_graph.invoke(initial_state)

    return {
        "question":     question,
        "route":        result.get("route"),
        "agents_used":  result.get("agents_used", []),
        "answer":       result.get("final_answer", ""),
    }