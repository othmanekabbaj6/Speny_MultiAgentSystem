import sys
from agents.budget_agent import run_budget_agent
from agents.goal_agent import run_goal_agent
from agents.anomaly_agent import run_anomaly_agent
from agents.behavioral_agent import run_behavioral_agent
from agents.persona_agent import run_persona_agent
from agents.forecast_agent import run_forecast_agent
from agents.simulation_agent import run_simulation_agent
from agents.retrieval_agent import run_retrieval_agent
from agents.advice_agent import run_advice_agent
from agents.explanation_agent import run_explanation_agent, explain_concept


USER_ID = "S6pwTrQB8R7GyuvBdyp0"

# ─── TESTS ────────────────────────────────────────────────────

def test_budget():
    print("\n" + "="*50)
    print("🤖 BUDGET AGENT — Avril 2026")
    print("="*50)
    result = run_budget_agent(
        USER_ID,
        question="Analyse mes dépenses du mois d'avril 2026. Y a-t-il des dépassements ?",
        month=4, year=2026
    )
    print(result['answer'])
    print(f"\n📊 Transactions : {result['month_transactions_count']}")
    print(f"💰 Dépenses par catégorie : {result['spending_by_category']}")

def test_goal():
    print("\n" + "="*50)
    print("🎯 GOAL AGENT — Avril 2026")
    print("="*50)
    result = run_goal_agent(
        USER_ID,
        question="Suis-je sur la bonne voie pour atteindre mes objectifs ?",
        month=4, year=2026
    )
    print(result['answer'])
    print(f"\n📊 Résumé mensuel : {result['monthly_summary']}")

def test_anomaly_global():
    print("\n" + "="*50)
    print("🔍 ANOMALY AGENT — Mode Global")
    print("="*50)
    result = run_anomaly_agent(USER_ID, mode="global")
    print(result['answer'])
    print(f"\n📊 Analysées : {result['total_transactions_analyzed']} | Anomalies : {result['anomalies_count']}")
    for a in result['anomalies']:
        print(f"  {a['anomaly_type']} | {a['date']} | {a['merchant']} | {a['amount']} MAD")

def test_anomaly_monthly():
    print("\n" + "="*50)
    print("🔍 ANOMALY AGENT — Avril 2026")
    print("="*50)
    result = run_anomaly_agent(USER_ID, mode="monthly", month=4, year=2026)
    print(result['answer'])
    print(f"\n📊 Anomalies en avril 2026 : {result['anomalies_count']}")
    for a in result['anomalies']:
        print(f"  {a['anomaly_type']} | {a['date']} | {a['merchant']} | {a['amount']} MAD")

def test_anomaly_all_months():
    print("\n" + "="*50)
    print("🔍 ANOMALY AGENT — Tous les mois")
    print("="*50)
    result = run_anomaly_agent(USER_ID, mode="all_months")
    for month_key, anomalies in result['monthly_results'].items():
        status = f"🚨 {len(anomalies)} anomalie(s)" if anomalies else "✅ Aucune"
        print(f"  {month_key} → {status}")
        for a in anomalies:
            print(f"    {a['anomaly_type']} | {a['merchant']} | {a['amount']} MAD")
    print(f"\n📊 Total : {result['anomalies_count']}")

def test_behavioral():
    print("\n" + "="*50)
    print("🧠 BEHAVIORAL AGENT")
    print("="*50)
    result = run_behavioral_agent(USER_ID)
    print(result['answer'])
    b = result['behavior']
    print(f"\n📅 Jour le plus dépensier : {b['day_analysis']['top_day']}")
    print(f"🏷️ Catégorie dominante : {b['category_analysis']['top_category']} ({b['category_analysis']['top_category_pct']}%)")
    print(f"📈 Tendance : {b['trend_analysis']['trend_direction']} ({b['trend_analysis']['trend_pct']}%)")
    print(f"🍽️ Food/semaine : {b['food_analysis']['avg_food_transactions_per_week']} transactions")

def test_persona():
    print("\n" + "="*50)
    print("👤 PERSONA AGENT")
    print("="*50)
    result = run_persona_agent(USER_ID)
    print(result['answer'])
    print(f"\n🧬 Type : {result['persona']['persona']}")
    print(f"📊 Score global : {result['persona']['global_score']}/100")
    print(f"💰 Taux épargne : {result['savings_ratio']}%")
    print(f"⚖️  Tolérance risque : {result['risk_tolerance']}")
    for k, v in result['scores'].items():
        print(f"   {k} : {v}/100")

def test_forecast():
    print("\n" + "="*50)
    print("📈 FORECAST AGENT")
    print("="*50)
    result = run_forecast_agent(USER_ID)
    print(result['answer'])
    print(f"\n💰 Total prévu : {result['total_predicted']} MAD")
    print(f"🚨 Catégories à risque : {len(result['at_risk_categories'])}")
    for f in result['forecasts']:
        print(f"  {f['category']:<15} | prédit: {f['predicted']} MAD | historique: {f['historical_mean']} MAD | tendance: {f['trend_pct']}%")

def test_simulation():
    print("\n" + "="*50)
    print("🎮 SIMULATION AGENT")
    print("="*50)

    print("\n--- Scénario 1 : Réduction Food 20% ---")
    result = run_simulation_agent(USER_ID, scenario="category_reduction", category="Food", reduction_pct=20, months=6)
    print(result['answer'])
    r = result['result']
    print(f"\n💰 Économie mensuelle : {r['reduction_amount']} MAD")
    print(f"💰 Total sur 6 mois : {r['total_saved_over_period']} MAD")

    print("\n--- Scénario 2 : Augmentation revenus 10% ---")
    result = run_simulation_agent(USER_ID, scenario="income_increase", income_increase_pct=10, months=6)
    print(result['answer'])
    r = result['result']
    print(f"\n💰 Gain mensuel : {r['increase_amount']} MAD")
    print(f"📈 Nouveau taux épargne : {r['new_savings_rate']}%")

    print("\n--- Scénario 3 : Objectif Trip Japan ---")
    result = run_simulation_agent(USER_ID, scenario="goal_achievement", goal_index=0, extra_monthly_savings=1000)
    print(result['answer'])
    r = result['result']
    print(f"\n🎯 Sans extra : {r['months_without_extra']} mois")
    print(f"🎯 Avec +1000 MAD/mois : {r['months_with_extra']} mois")

def test_retrieval():
    print("\n" + "="*50)
    print("🔍 RETRIEVAL AGENT")
    print("="*50)
    questions = [
        "Combien j'ai dépensé chez Marjane ?",
        "Quelles sont mes dépenses Transport ce mois-ci ?",
        "Quel est mon objectif le plus proche d'être atteint ?",
        "Est-ce que j'ai dépassé mon budget Food ?",
    ]
    for q in questions:
        print(f"\n❓ {q}")
        result = run_retrieval_agent(USER_ID, q)
        print(f"💬 {result['answer']}")
        print(f"   ({result['docs_found']} documents récupérés)")

def test_advice():
    print("\n" + "="*50)
    print("🧠 ADVICE AGENT (Orchestrateur)")
    print("="*50)
    result = run_advice_agent(
        USER_ID,
        mode="full",
        month=4,
        year=2026,
        question="Donne-moi une analyse complète de ma situation financière et tes meilleures recommandations."
    )
    print(result['answer'])
    print("\n" + "-"*40)
    print("📊 Résumés extraits :")
    for k, v in result['summaries'].items():
        if v:
            print(f"\n[{k.upper()}]\n{v}")

def test_advice_debug():
    print("\n=== DEBUG KEYS ===")
    r = run_budget_agent(USER_ID)
    print(f"\nBudget keys: {list(r.keys())}")
    print(f"Budget sample: {str(r)[:300]}")
    r = run_goal_agent(USER_ID)
    print(f"\nGoal keys: {list(r.keys())}")
    print(f"Goal sample: {str(r)[:300]}")
    r = run_anomaly_agent(USER_ID, mode="monthly")
    print(f"\nAnomaly keys: {list(r.keys())}")
    print(f"Anomaly sample: {str(r)[:300]}")


def test_explanation():
    print("\n" + "="*50)
    print("🗣️ EXPLANATION AGENT")
    print("="*50)

    # Test 1 : Explique le résultat du Behavioral Agent
    print("\n--- Explication Behavioral (grand public) ---")
    behavioral_result = run_behavioral_agent(USER_ID)
    expl = run_explanation_agent(
        agent_name="behavioral_agent",
        agent_result=behavioral_result,
        question="Quelles sont mes habitudes de dépenses ?",
        audience="grand public"
    )
    print(expl['explanation'])

    # Test 2 : Explique le résultat du Forecast Agent
    print("\n--- Explication Forecast (débutant) ---")
    forecast_result = run_forecast_agent(USER_ID)
    expl = run_explanation_agent(
        agent_name="forecast_agent",
        agent_result=forecast_result,
        question="Combien vais-je dépenser le mois prochain ?",
        audience="débutant"
    )
    print(expl['explanation'])

    # Test 3 : Explique un concept
    print("\n--- Explication concept : taux d'épargne ---")
    concept_expl = explain_concept(
        "taux d'épargne",
        "je gagne 24000 MAD/mois et dépense 18000 MAD"
    )
    print(concept_expl)

def test_retrieval_react():
    print("\n" + "="*50)
    print("🔍 RETRIEVAL AGENT — ReAct")
    print("="*50)
    from agents.retrieval_agent import run_retrieval_agent_react
    result = run_retrieval_agent_react(
        USER_ID,
        "Est-ce que j'ai dépassé mon budget Food ce mois-ci ?"
    )
    print(result['answer'])


def test_graph():
    from orchestration.graph import run_graph
    print("\n" + "="*50)
    print("🕸️  LANGGRAPH — Orchestration conditionnelle")
    print("="*50)

    questions = [
        "Est-ce que j'ai dépassé mon budget Food ?",
        "Quelles sont mes prévisions pour le mois prochain ?",
        "Y a-t-il des dépenses inhabituelles ?",
        "Donne-moi une analyse complète de mes finances",
    ]

    for q in questions:
        print(f"\n❓ {q}")
        result = run_graph("S6pwTrQB8R7GyuvBdyp0", q, month=4, year=2026)
        print(f"🔀 Route : {result['route']}")
        print(f"🤖 Agents : {result['agents_used']}")
        print(f"💬 {result['answer'][:200]}...")


def test_memory():
    from memory.short_term import session_manager
    from memory.long_term import (
        save_preferences, get_preferences,
        save_advice_to_history, get_advice_history,
        save_user_note, get_user_notes,
        get_memory_context_for_llm
    )

    print("\n" + "="*50)
    print("🧠 MEMORY — Court terme + Long terme")
    print("="*50)

    # ── Short term ──
    print("\n--- Mémoire court terme ---")
    session = session_manager.get_session(USER_ID)
    session.add_user_message("Comment vont mes finances ?")
    session.add_assistant_message("Food dépassé à 185%...")
    session.add_user_message("Et si je réduis Food de 20% ?")
    print(f"Messages en session : {len(session.get_history())}")
    print(f"Historique :\n{session.get_history_as_text()}")

    # ── Long term ──
    print("\n--- Mémoire long terme (Firestore) ---")
    save_preferences(USER_ID, {"mode": "quick", "audience": "grand public"})
    prefs = get_preferences(USER_ID)
    print(f"Préférences : {prefs}")

    save_advice_to_history(USER_ID, "Analyse complète ?", "Food dépassé...", ["budget", "goal"])
    history = get_advice_history(USER_ID)
    print(f"Historique conseils : {len(history)} entrée(s)")

    save_user_note(USER_ID, "Je veux économiser pour une voiture en décembre")
    notes = get_user_notes(USER_ID)
    print(f"Notes : {len(notes)} note(s)")

    print(f"\nContexte LLM :\n{get_memory_context_for_llm(USER_ID)}")

# ─── MAPPING ──────────────────────────────────────────────────

AGENTS = {
    "budget":         test_budget,
    "goal":           test_goal,
    "anomaly":        test_anomaly_global,
    "anomaly:monthly": test_anomaly_monthly,
    "anomaly:all":    test_anomaly_all_months,
    "behavioral":     test_behavioral,
    "persona":        test_persona,
    "forecast":       test_forecast,
    "simulation":     test_simulation,
    "retrieval":      test_retrieval,
    "advice":         test_advice,
    "advice_debug":   test_advice_debug,
    "explanation":    test_explanation,
    "retrieval_react": test_retrieval_react,
    "graph": test_graph,
    "memory": test_memory,
    "all":            None
}

if __name__ == "__main__":
    arg = sys.argv[1] if len(sys.argv) > 1 else "all"

    if arg not in AGENTS:
        print(f"❌ Agent inconnu : '{arg}'")
        print(f"   Agents disponibles : {list(AGENTS.keys())}")
        sys.exit(1)

    if arg == "all":
        test_budget()
        test_goal()
        test_anomaly_global()
        test_anomaly_monthly()
        test_anomaly_all_months()
        test_behavioral()
        test_persona()
        test_forecast()
        test_simulation()
        test_retrieval()
        test_advice()
    else:
        AGENTS[arg]()

    print("\n✅ Done !")