from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import logging

from agents.budget_agent import run_budget_agent
from agents.goal_agent import run_goal_agent
from agents.anomaly_agent import run_anomaly_agent
from agents.behavioral_agent import run_behavioral_agent
from agents.persona_agent import run_persona_agent
from agents.forecast_agent import run_forecast_agent
from agents.simulation_agent import run_simulation_agent
from agents.retrieval_agent import run_retrieval_agent, run_retrieval_agent_react
from agents.advice_agent import run_advice_agent
from agents.explanation_agent import run_explanation_agent, explain_concept

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="SpendWise API",
    description="Système multi-agents IA de gestion financière personnelle",
    version="1.0.0"
)

# ─── CORS (pour appels depuis une interface web/mobile) ────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Modèles de requêtes ───────────────────────────────────────

class AdviceRequest(BaseModel):
    user_id: str
    question: Optional[str] = None
    mode: Optional[str] = "full"        # full | quick | forecast | profile
    month: Optional[int] = None
    year: Optional[int] = None

class ChatRequest(BaseModel):
    user_id: str
    question: str
    use_react: Optional[bool] = True    # True = ReAct, False = RAG simple

class SimulationRequest(BaseModel):
    user_id: str
    scenario: str                        # category_reduction | income_increase | goal_achievement
    category: Optional[str] = None
    reduction_pct: Optional[float] = 20
    income_increase_pct: Optional[float] = 10
    goal_index: Optional[int] = 0
    extra_monthly_savings: Optional[float] = 0
    months: Optional[int] = 6

class ExplainRequest(BaseModel):
    user_id: str
    agent_name: str
    question: Optional[str] = None
    audience: Optional[str] = "grand public"  # grand public | expert | débutant

class ConceptRequest(BaseModel):
    concept: str
    context: Optional[str] = ""

class BudgetRequest(BaseModel):
    user_id: str
    month: Optional[int] = None
    year: Optional[int] = None

class AnomalyRequest(BaseModel):
    user_id: str
    mode: Optional[str] = "monthly"    # monthly | global | all_months
    month: Optional[int] = None
    year: Optional[int] = None

# ─── Health check ──────────────────────────────────────────────

@app.get("/")
def root():
    return {
        "app": "SpendWise API",
        "version": "1.0.0",
        "status": "running",
        "endpoints": [
            "GET  /health",
            "POST /api/advice",
            "POST /api/chat",
            "GET  /api/budget/{user_id}",
            "GET  /api/goals/{user_id}",
            "GET  /api/anomalies/{user_id}",
            "GET  /api/behavioral/{user_id}",
            "GET  /api/persona/{user_id}",
            "GET  /api/forecast/{user_id}",
            "POST /api/simulate",
            "POST /api/explain",
            "POST /api/explain/concept",
        ]
    }

@app.get("/health")
def health():
    return {"status": "ok"}

# ─── Advice (orchestrateur central) ───────────────────────────

@app.post("/api/advice")
def advice(req: AdviceRequest):
    """
    Analyse financière complète via l'Advice Agent (orchestrateur).
    mode: full (~90s) | quick (~20s) | forecast (~30s) | profile (~15s)
    """
    try:
        result = run_advice_agent(
            user_id=req.user_id,
            question=req.question,
            mode=req.mode,
            month=req.month,
            year=req.year
        )
        return {
            "success": True,
            "agent": "advice_agent",
            "mode": result["mode"],
            "answer": result["answer"],
            "summaries": result["summaries"]
        }
    except Exception as e:
        logger.error(f"Advice agent error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ─── Chat (Retrieval Agent + ReAct) ───────────────────────────

@app.post("/api/chat")
def chat(req: ChatRequest):
    """
    Question en langage naturel sur les données financières.
    use_react=True  → ReAct (Thought → Action → Observation → Answer)
    use_react=False → RAG simple
    """
    try:
        if req.use_react:
            result = run_retrieval_agent_react(req.user_id, req.question)
        else:
            result = run_retrieval_agent(req.user_id, req.question)
        return {
            "success": True,
            "agent": result["agent"],
            "method": result.get("method", "RAG"),
            "question": req.question,
            "answer": result["answer"],
            "tool_used": result.get("tool_used", None),
            "observation": result.get("observation", None),
        }
    except Exception as e:
        logger.error(f"Chat error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ─── Budget ────────────────────────────────────────────────────

@app.get("/api/budget/{user_id}")
def budget(user_id: str, month: Optional[int] = None, year: Optional[int] = None):
    """Analyse du budget du mois."""
    try:
        result = run_budget_agent(user_id, month=month, year=year)
        return {
            "success": True,
            "answer": result["answer"],
            "budgets": result["budgets"],
            "spending_by_category": result["spending_by_category"],
            "spending_no_budget": result["spending_no_budget"],
            "month_transactions_count": result["month_transactions_count"],
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ─── Goals ─────────────────────────────────────────────────────

@app.get("/api/goals/{user_id}")
def goals(user_id: str, month: Optional[int] = None, year: Optional[int] = None):
    """Analyse des objectifs financiers."""
    try:
        result = run_goal_agent(user_id, month=month, year=year)
        return {
            "success": True,
            "answer": result["answer"],
            "goals": result["goals"],
            "monthly_summary": result["monthly_summary"],
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ─── Anomalies ─────────────────────────────────────────────────

@app.post("/api/anomalies")
def anomalies(req: AnomalyRequest):
    """Détection d'anomalies dans les transactions."""
    try:
        result = run_anomaly_agent(
            req.user_id,
            mode=req.mode,
            month=req.month,
            year=req.year
        )
        return {
            "success": True,
            "answer": result["answer"],
            "anomalies": result["anomalies"],
            "anomalies_count": result["anomalies_count"],
            "total_transactions_analyzed": result["total_transactions_analyzed"],
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ─── Behavioral ────────────────────────────────────────────────

@app.get("/api/behavioral/{user_id}")
def behavioral(user_id: str):
    """Analyse des habitudes de dépenses."""
    try:
        result = run_behavioral_agent(user_id)
        return {
            "success": True,
            "answer": result["answer"],
            "behavior": result["behavior"],
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ─── Persona ───────────────────────────────────────────────────

@app.get("/api/persona/{user_id}")
def persona(user_id: str):
    """Profil financier de l'utilisateur."""
    try:
        result = run_persona_agent(user_id)
        return {
            "success": True,
            "answer": result["answer"],
            "persona": result["persona"],
            "scores": result["scores"],
            "savings_ratio": result["savings_ratio"],
            "risk_tolerance": result["risk_tolerance"],
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ─── Forecast ──────────────────────────────────────────────────

@app.get("/api/forecast/{user_id}")
def forecast(user_id: str):
    """Prévisions des dépenses du mois prochain (LSTM)."""
    try:
        result = run_forecast_agent(user_id)
        return {
            "success": True,
            "answer": result["answer"],
            "forecasts": result["forecasts"],
            "total_predicted": result["total_predicted"],
            "at_risk_categories": result["at_risk_categories"],
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ─── Simulation ────────────────────────────────────────────────

@app.post("/api/simulate")
def simulate(req: SimulationRequest):
    """
    Simulation de scénarios financiers.
    scenario: category_reduction | income_increase | goal_achievement
    """
    try:
        result = run_simulation_agent(
            user_id=req.user_id,
            scenario=req.scenario,
            category=req.category,
            reduction_pct=req.reduction_pct,
            income_increase_pct=req.income_increase_pct,
            goal_index=req.goal_index,
            extra_monthly_savings=req.extra_monthly_savings,
            months=req.months
        )
        return {
            "success": True,
            "answer": result["answer"],
            "scenario": req.scenario,
            "result": result["result"],
            "baseline": result["baseline"],
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ─── Explanation ───────────────────────────────────────────────

@app.post("/api/explain")
def explain(req: ExplainRequest):
    """
    Traduit l'output d'un agent en langage naturel simple.
    audience: grand public | expert | débutant
    """
    try:
        # Appelle l'agent source pour avoir son résultat
        agent_runners = {
            "budget_agent":    lambda: run_budget_agent(req.user_id),
            "goal_agent":      lambda: run_goal_agent(req.user_id),
            "behavioral_agent":lambda: run_behavioral_agent(req.user_id),
            "persona_agent":   lambda: run_persona_agent(req.user_id),
            "forecast_agent":  lambda: run_forecast_agent(req.user_id),
        }
        if req.agent_name not in agent_runners:
            raise HTTPException(
                status_code=400,
                detail=f"Agent inconnu : {req.agent_name}. "
                       f"Disponibles : {list(agent_runners.keys())}"
            )
        agent_result = agent_runners[req.agent_name]()
        result = run_explanation_agent(
            agent_name=req.agent_name,
            agent_result=agent_result,
            question=req.question,
            audience=req.audience
        )
        return {
            "success": True,
            "agent_name": req.agent_name,
            "audience": req.audience,
            "explanation": result["explanation"],
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/explain/concept")
def explain_concept_endpoint(req: ConceptRequest):
    """Explique un concept financier en langage simple."""
    try:
        explanation = explain_concept(req.concept, req.context)
        return {
            "success": True,
            "concept": req.concept,
            "explanation": explanation,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))