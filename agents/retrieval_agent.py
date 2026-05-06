from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from rag.query_engine import search_user_data
from data.firebase_client import get_transactions, get_budgets, get_goals
from config.settings import settings
from datetime import datetime, timezone
import logging

logger = logging.getLogger(__name__)

llm = ChatGroq(
    api_key=settings.groq_api_key,
    model_name=settings.llm_model
)

retrieval_prompt = PromptTemplate(
    input_variables=["question", "documents", "metadata"],
    template="""
Tu es un assistant financier intelligent avec accès aux données financières de l'utilisateur.
Réponds en français, de manière précise et concise.

DOCUMENTS RÉCUPÉRÉS (extraits de tes données financières) :
{documents}

MÉTADONNÉES CONTEXTUELLES :
{metadata}

QUESTION : {question}

Instructions :
- Réponds directement à la question en te basant sur les documents fournis
- Cite des chiffres précis quand c'est pertinent
- Si la réponse n'est pas dans les documents, dis-le clairement
- Sois concis : 3-5 phrases maximum sauf si une liste est nécessaire
- Toujours répondre en MAD (dirhams marocains)
"""
)

# ─── Enrichissement du contexte ───────────────────────────────
def build_metadata_context(user_id: str, question: str) -> str:
    lines = []
    now = datetime.now(timezone.utc)

    try:
        transactions = get_transactions(user_id, limit=200)
        recent = []
        for t in transactions:
            date = t.get('date')
            if date:
                if hasattr(date, 'tzinfo') and date.tzinfo is None:
                    date = date.replace(tzinfo=timezone.utc)
                if (now - date).days <= 30:
                    recent.append(t)

        income_30d   = sum(t.get('amount', 0) for t in recent if t.get('type') == 'income')
        expenses_30d = sum(t.get('amount', 0) for t in recent if t.get('type') == 'expense')

        lines.append("Période : 30 derniers jours")
        lines.append(f"Revenus : {round(income_30d, 2)} MAD")
        lines.append(f"Dépenses : {round(expenses_30d, 2)} MAD")
        lines.append(f"Solde net : {round(income_30d - expenses_30d, 2)} MAD")

        budgets = get_budgets(user_id)
        if budgets:
            lines.append(f"Budgets actifs : {len(budgets)}")
            for b in budgets:
                lines.append(f"  - {b.get('category')} : {b.get('limit_amount')} MAD/mois")

        goals = get_goals(user_id)
        active_goals = [g for g in goals if g.get('status') == 'active']
        if active_goals:
            lines.append(f"Objectifs actifs : {len(active_goals)}")
            for g in active_goals:
                lines.append(f"  - {g.get('title')} : {g.get('progress_pct', 0):.1f}% complété")

    except Exception as e:
        logger.warning(f"Erreur metadata context : {e}")
        lines.append("Métadonnées non disponibles.")

    return "\n".join(lines)


# ─── Recherche et réponse standard ────────────────────────────
def run_retrieval_agent(user_id: str, question: str, top_k: int = 8) -> dict:
    docs = search_user_data(user_id, question, top_k=top_k)

    if not docs:
        return {
            "agent":     "retrieval_agent",
            "question":  question,
            "answer":    "Aucune donnée trouvée pour répondre à cette question.",
            "docs_found": 0
        }

    documents_str = "\n\n".join([
        f"[Doc {i+1}] {d['text']}"
        for i, d in enumerate(docs)
    ])

    metadata_str = build_metadata_context(user_id, question)

    chain    = retrieval_prompt | llm
    response = chain.invoke({
        "question":  question,
        "documents": documents_str,
        "metadata":  metadata_str
    })

    return {
        "agent":     "retrieval_agent",
        "question":  question,
        "answer":    response.content,
        "docs_found": len(docs),
        "sources":   [d['text'][:80] + "..." for d in docs[:3]]
    }


# ─── Outils internes pour ReAct ───────────────────────────────
def _make_tools(user_id: str) -> dict:

    # Outil 1 — recherche sémantique RAG
    def rag_search(query: str) -> str:
        docs = search_user_data(user_id, query, top_k=6)
        return "\n".join([d['text'] for d in docs]) if docs else "Aucun résultat."

    # Outil 2 — résumé financier des 30 derniers jours
    def get_financial_summary(query: str = "") -> str:
        transactions = get_transactions(user_id, limit=100)
        now = datetime.now(timezone.utc)
        recent = []
        for t in transactions:
            date = t.get('date')
            if date:
                if hasattr(date, 'tzinfo') and date.tzinfo is None:
                    date = date.replace(tzinfo=timezone.utc)
                if (now - date).days <= 30:
                    recent.append(t)
        expenses = sum(t.get('amount', 0) for t in recent if t.get('type') == 'expense')
        income   = sum(t.get('amount', 0) for t in recent if t.get('type') == 'income')
        by_cat   = {}
        for t in recent:
            if t.get('type') == 'expense':
                cat = t.get('category', 'Other')
                by_cat[cat] = by_cat.get(cat, 0) + t.get('amount', 0)
        top_cats = sorted(by_cat.items(), key=lambda x: -x[1])[:3]
        cats_str = ", ".join([f"{c}: {round(v, 0)} MAD" for c, v in top_cats])
        return (
            f"Revenus 30j: {round(income, 2)} MAD | "
            f"Dépenses 30j: {round(expenses, 2)} MAD | "
            f"Solde: {round(income - expenses, 2)} MAD | "
            f"Top catégories: {cats_str}"
        )

    # Outil 3 — statut réel des budgets (appelle le Budget Agent)
    def get_budget_status(query: str = "") -> str:
        from agents.budget_agent import run_budget_agent
        now = datetime.now(timezone.utc)
        # On prend le mois précédent car le mois courant est souvent vide
        if now.month == 1:
            month, year = 12, now.year - 1
        else:
            month, year = now.month - 1, now.year

        result   = run_budget_agent(user_id, month=month, year=year)
        budgets  = result.get('budgets', [])
        spending = result.get('spending_by_category', {})

        if not budgets:
            return "Aucun budget défini."

        lines = [f"Statut des budgets pour {month:02d}/{year} :"]
        for b in budgets:
            cat   = b.get('category', '')
            limit = b.get('limit_amount', 0)
            spent = next(
                (v for k, v in spending.items() if k.lower() == cat.lower()), 0
            )
            pct    = round(spent / limit * 100, 1) if limit > 0 else 0
            status = "⚠️ DÉPASSÉ" if pct > 100 else "✅ OK"
            lines.append(
                f"  - {cat} : {round(spent, 2)} MAD / {limit} MAD "
                f"({pct}%) {status}"
            )

        # Ajoute aussi les dépenses sans budget défini
        spending_no_budget = result.get('spending_no_budget', {})
        if spending_no_budget:
            top = sorted(spending_no_budget.items(), key=lambda x: -x[1])[:3]
            lines.append("Sans budget défini :")
            for cat, amt in top:
                lines.append(f"  - {cat} : {round(amt, 2)} MAD")

        return "\n".join(lines)

    return {
        "RechercheRAG":    rag_search,
        "ResumeFinancier": get_financial_summary,
        "BudgetStatus":    get_budget_status,
    }


# ─── Version ReAct manuelle ───────────────────────────────────
def run_retrieval_agent_react(user_id: str, question: str) -> dict:
    """
    Version ReAct du Retrieval Agent.
    Implémente Thought → Action → Observation → Final Answer
    manuellement, sans create_react_agent.
    Compatible LangChain 1.x / LangGraph.
    """
    tools_map = _make_tools(user_id)

    # ── Étape 1 : raisonnement + choix d'outil ──
    step1_prompt = PromptTemplate.from_template(
        """Tu es un assistant financier intelligent. Réponds TOUJOURS en français.

Outils disponibles :
- RechercheRAG      : recherche sémantique dans les transactions, budgets et objectifs. Input: mot-clé ou question
- ResumeFinancier   : résumé financier des 30 derniers jours (revenus, dépenses, top catégories). Input: n'importe quel texte
- BudgetStatus      : vérifie si un budget est dépassé ce mois-ci avec les chiffres exacts. Input: catégorie concernée (ex: Food, Education)

Réponds avec EXACTEMENT ce format (une ligne par champ, sans texte supplémentaire) :
Thought: [pourquoi et comment tu vas répondre]
Action: [RechercheRAG ou ResumeFinancier ou BudgetStatus]
Action Input: [l'input exact à passer à l'outil]

Question: {question}
Thought:"""
    )

    step1 = (step1_prompt | llm).invoke({"question": question})
    raw   = step1.content
    logger.info(f"ReAct Step 1:\n{raw}")

    # Parse Action / Action Input
    action_used  = ""
    action_input = ""
    for line in raw.split("\n"):
        line = line.strip()
        if line.startswith("Action:"):
            action_used  = line.replace("Action:", "").strip()
        elif line.startswith("Action Input:"):
            action_input = line.replace("Action Input:", "").strip()

    # ── Étape 2 : exécution de l'outil ──
    if action_used in tools_map:
        try:
            observation = tools_map[action_used](action_input)
        except Exception as e:
            observation = f"Erreur outil: {e}"
    else:
        # Fallback : RAG + résumé financier
        rag_obs = tools_map["RechercheRAG"](question)
        fin_obs = tools_map["ResumeFinancier"]("")
        observation = f"RAG:\n{rag_obs}\n\nRésumé financier:\n{fin_obs}"
        action_used = "RechercheRAG + ResumeFinancier (fallback)"

    logger.info(f"Observation ({action_used}): {observation[:200]}")

    # ── Étape 3 : réponse finale ──
    step2_prompt = PromptTemplate.from_template(
        """Tu es un assistant financier. Réponds en français, de façon concise et précise.

Question: {question}

Raisonnement:
{reasoning}

Observation de l'outil ({action}):
{observation}

Thought: J'ai les informations nécessaires pour répondre.
Final Answer:"""
    )

    step2 = (step2_prompt | llm).invoke({
        "question":    question,
        "reasoning":   raw,
        "action":      action_used,
        "observation": observation,
    })

    return {
        "agent":       "retrieval_agent_react",
        "method":      "ReAct (manuel, compatible LangChain 1.x)",
        "question":    question,
        "reasoning":   raw,
        "tool_used":   action_used,
        "tool_input":  action_input,
        "observation": observation,
        "answer":      step2.content,
    }