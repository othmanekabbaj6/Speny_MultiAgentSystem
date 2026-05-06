import numpy as np
import pandas as pd
from collections import Counter, defaultdict
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import LabelEncoder
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

anomaly_prompt = PromptTemplate(
    input_variables=["anomalies", "period", "context", "question"],
    template="""
Tu es un assistant financier bienveillant qui aide l'utilisateur à mieux comprendre ses habitudes de dépenses.
L'utilisateur saisit lui-même ses transactions dans une application de gestion financière.
Réponds en français.

PÉRIODE ANALYSÉE : {period}

DÉPENSES INHABITUELLES DÉTECTÉES :
{anomalies}

CONTEXTE SUPPLÉMENTAIRE :
{context}

QUESTION : {question}

Pour chaque dépense inhabituelle :
- Explique pourquoi elle sort de l'ordinaire par rapport aux habitudes de l'utilisateur
- Aide l'utilisateur à prendre conscience de cette dépense
- Suggère comment éviter ce type de dépense excessive à l'avenir
- Si c'est une possible erreur de saisie, demande à l'utilisateur de vérifier

Ne parle jamais de fraude ou de contestation bancaire.
Sois encourageant et constructif, pas accusateur.
"""
)

# ─── Préparation des données ───────────────────────────────────
def prepare_features(transactions: list[dict]) -> pd.DataFrame:
    rows = []
    for t in transactions:
        date = t.get('date')
        if hasattr(date, 'tzinfo') and date.tzinfo is None:
            date = date.replace(tzinfo=timezone.utc)

        rows.append({
            'amount': t.get('amount', 0),
            'hour': date.hour if date else 12,
            'day_of_week': date.weekday() if date else 0,
            'day_of_month': date.day if date else 1,
            'month': date.month if date else 1,
            'year': date.year if date else 2025,
            'category': t.get('category', 'Other'),
            'type': t.get('type', 'expense'),
            'id': t.get('id', ''),
            'merchant_name': t.get('merchant_name', ''),
            'description': t.get('description', ''),
            'date_str': date.strftime('%Y-%m-%d') if date else '',
            'currency': t.get('currency', 'MAD'),
        })

    return pd.DataFrame(rows)

def encode_features(df: pd.DataFrame) -> np.ndarray:
    le_cat = LabelEncoder()
    le_type = LabelEncoder()

    features = np.column_stack([
        df['amount'].values,
        df['hour'].values,
        df['day_of_week'].values,
        df['day_of_month'].values,
        le_cat.fit_transform(df['category'].values),
        le_type.fit_transform(df['type'].values),
    ])
    return features

# ─── Stats mensuelles ─────────────────────────────────────────
def get_monthly_category_stats(df: pd.DataFrame) -> dict:
    stats = {}
    grouped = df.groupby(['year', 'month', 'category'])['amount']
    for (year, month, category), group in grouped:
        stats[(int(year), int(month), category)] = {
            'mean': group.mean(),
            'std': group.std() if len(group) > 1 else group.mean() * 0.1
        }
    return stats

def calculate_deviation(row, monthly_stats: dict, global_df: pd.DataFrame) -> tuple:
    year = int(row['year'])
    month = int(row['month'])
    category = row['category']
    amount = row['amount']

    monthly_key = (year, month, category)

    if monthly_key in monthly_stats:
        stat = monthly_stats[monthly_key]
        mean = stat['mean']
        std = stat['std'] if stat['std'] > 0 else mean * 0.1
        deviation = abs(amount - mean) / (std + 1e-9)
        context = f"moyenne {month:02d}/{year}"
    else:
        cat_amounts = global_df[global_df['category'] == category]['amount']
        mean = cat_amounts.mean()
        std = cat_amounts.std() if len(cat_amounts) > 1 else mean * 0.1
        deviation = abs(amount - mean) / (std + 1e-9)
        context = "moyenne globale"

    return round(deviation, 2), round(mean, 2), context

# ─── Marchands récurrents ─────────────────────────────────────
def get_recurring_merchants(transactions: list[dict], threshold: int = 6) -> set:
    """
    Un marchand est récurrent seulement si :
    1. Il apparaît threshold+ fois
    2. ET ses montants sont stables (coefficient de variation < 30%)

    Un marchand fréquent mais avec des montants très variables
    (ex: Electroplanet) n'est PAS filtré car ses anomalies sont réelles.
    """
    merchant_amounts = defaultdict(list)
    for t in transactions:
        merchant = t.get('merchant_name', '').lower().strip()
        merchant_amounts[merchant].append(t.get('amount', 0))

    recurring = set()
    for merchant, amounts in merchant_amounts.items():
        if len(amounts) < threshold:
            continue

        mean = np.mean(amounts)
        std = np.std(amounts)
        cv = std / mean if mean > 0 else 1.0

        if cv < 0.30:
            recurring.add(merchant)
            logger.info(f"Récurrent stable : {merchant} ({len(amounts)}x, CV={cv:.2f})")
        else:
            logger.info(f"Fréquent mais variable : {merchant} ({len(amounts)}x, CV={cv:.2f}) → gardé")

    return recurring

# ─── Classification ───────────────────────────────────────────
def classify_anomaly(deviation: float, score: float) -> str:
    if deviation > 3:
        return "⚠️ Dépense très inhabituelle"
    elif deviation > 1.5:
        return "📊 Dépense au-dessus de la normale"
    elif score < -0.15:
        return "🔍 Possible erreur de saisie"
    else:
        return "💡 Dépense impulsive"

# ─── Core : Isolation Forest ──────────────────────────────────
def _run_isolation_forest(
    df: pd.DataFrame,
    all_df: pd.DataFrame,
    monthly_stats: dict,
    contamination: float
) -> list[dict]:
    """
    Lance Isolation Forest sur df.
    Les marchands récurrents sont déjà exclus avant appel.
    Filtre les faux positifs (déviation 0.0x, déviation trop faible).
    """
    if len(df) < 10:
        logger.warning("Pas assez de transactions pour l'analyse (minimum 10)")
        return []

    X = encode_features(df)

    model = IsolationForest(
        contamination=contamination,
        random_state=42,
        n_estimators=100
    )
    predictions = model.fit_predict(X)
    scores = model.score_samples(X)

    anomalies = []
    for i, (pred, score) in enumerate(zip(predictions, scores)):
        if pred == -1:
            row = df.iloc[i]
            deviation, monthly_mean, context_label = calculate_deviation(
                row, monthly_stats, all_df
            )

            # Filtre faux positifs :
            # 1. déviation 0.0 = seule transaction du mois dans cette catégorie
            if deviation == 0.0:
                continue

            # 2. déviation trop faible avec score pas très bas
            if deviation < 0.5 and score > -0.60:
                continue

            anomaly_type = classify_anomaly(deviation, score)

            anomalies.append({
                'id': row['id'],
                'date': row['date_str'],
                'month': f"{int(row['month']):02d}/{int(row['year'])}",
                'merchant': row['merchant_name'],
                'description': row['description'],
                'category': row['category'],
                'amount': round(row['amount'], 2),
                'currency': row['currency'],
                'anomaly_score': round(score, 4),
                'deviation_from_mean': deviation,
                'monthly_mean': monthly_mean,
                'context_label': context_label,
                'anomaly_type': anomaly_type,
            })

    anomalies.sort(key=lambda x: x['anomaly_score'])
    return anomalies

# ─── Mode global ──────────────────────────────────────────────
def detect_anomalies_global(
    transactions: list[dict],
    contamination: float = 0.05
) -> list[dict]:
    """
    Analyse toutes les transactions en une seule passe.
    Exclut les marchands récurrents stables AVANT l'entraînement.
    """
    if len(transactions) < 10:
        return []

    recurring = get_recurring_merchants(transactions, threshold=6)

    filtered = [
        t for t in transactions
        if t.get('merchant_name', '').lower().strip() not in recurring
    ]

    if len(filtered) < 10:
        logger.warning("Pas assez après filtrage, analyse sur toutes")
        filtered = transactions

    df = prepare_features(filtered)
    all_df = df.copy()
    monthly_stats = get_monthly_category_stats(all_df)

    return _run_isolation_forest(df, all_df, monthly_stats, contamination)

# ─── Mode mensuel ─────────────────────────────────────────────
def detect_anomalies_monthly(
    transactions: list[dict],
    month: int,
    year: int,
    contamination: float = 0.05
) -> list[dict]:
    """
    Analyse les transactions d'un mois précis.
    Exclut les marchands récurrents stables AVANT l'entraînement.
    Compare aux stats globales pour contextualiser.
    """
    recurring = get_recurring_merchants(transactions, threshold=6)

    all_filtered = [
        t for t in transactions
        if t.get('merchant_name', '').lower().strip() not in recurring
    ]
    all_df = prepare_features(all_filtered) if all_filtered else prepare_features(transactions)
    monthly_stats = get_monthly_category_stats(all_df)

    month_txs = [
        t for t in transactions
        if hasattr(t.get('date'), 'month')
        and t['date'].month == month
        and t['date'].year == year
        and t.get('merchant_name', '').lower().strip() not in recurring
    ]

    if len(month_txs) < 5:
        logger.warning(f"Pas assez de transactions non récurrentes pour {month:02d}/{year}")
        return []

    month_df = prepare_features(month_txs)

    return _run_isolation_forest(month_df, all_df, monthly_stats, contamination)

# ─── Mode tous les mois ───────────────────────────────────────
def detect_anomalies_all_months(
    transactions: list[dict],
    contamination: float = 0.05
) -> dict:
    """
    Lance l'analyse pour chaque mois disponible.
    Retourne : { '04/2026': [...anomalies], '03/2026': [...] }
    """
    months_available = set()
    for t in transactions:
        date = t.get('date')
        if hasattr(date, 'month'):
            months_available.add((date.month, date.year))

    results = {}
    for month, year in sorted(months_available, reverse=True):
        anomalies = detect_anomalies_monthly(
            transactions, month, year, contamination
        )
        key = f"{month:02d}/{year}"
        results[key] = anomalies
        logger.info(f"{key} → {len(anomalies)} anomalie(s)")

    return results

# ─── Formatage ────────────────────────────────────────────────
def format_anomalies(anomalies: list[dict]) -> str:
    if not anomalies:
        return "Aucune dépense inhabituelle détectée."

    lines = []
    for a in anomalies:
        lines.append(
            f"- [{a['anomaly_type']}] {a['date']} | {a['merchant']} | "
            f"{a['category']} | {a['amount']} {a['currency']} | "
            f"Moyenne ({a['context_label']}): {a['monthly_mean']} {a['currency']} | "
            f"Déviation: {a['deviation_from_mean']}x"
        )
    return "\n".join(lines)

def format_monthly_summary(monthly_results: dict) -> str:
    lines = []
    for month_key, anomalies in monthly_results.items():
        if anomalies:
            lines.append(f"\n📅 {month_key} — {len(anomalies)} dépense(s) inhabituelle(s) :")
            lines.append(format_anomalies(anomalies))
        else:
            lines.append(f"\n📅 {month_key} — ✅ Aucune anomalie")
    return "\n".join(lines) if lines else "Aucune anomalie détectée."

# ─── Agent principal ───────────────────────────────────────────
def run_anomaly_agent(
    user_id: str,
    question: str = None,
    month: int = None,
    year: int = None,
    mode: str = "global"
) -> dict:
    """
    3 modes :
    - "global"     → analyse tout l'historique
    - "monthly"    → analyse un mois précis (month + year requis)
    - "all_months" → analyse chaque mois séparément
    """
    if question is None:
        question = "Y a-t-il des dépenses inhabituelles dans mes transactions ?"

    # 1. Récupère toutes les dépenses
    transactions = get_transactions(user_id, limit=500)
    expenses = [t for t in transactions if t.get('type') == 'expense']

    # 2. Détection selon le mode
    anomalies = []
    monthly_results = {}
    period = ""

    if mode == "monthly" and month and year:
        anomalies = detect_anomalies_monthly(expenses, month, year)
        period = f"{month:02d}/{year}"

    elif mode == "all_months":
        monthly_results = detect_anomalies_all_months(expenses)
        anomalies = [a for month_list in monthly_results.values() for a in month_list]
        period = "Tous les mois disponibles"

    else:
        anomalies = detect_anomalies_global(expenses)
        period = "Historique complet"

    # 3. Contexte RAG
    context_docs = search_user_data(user_id, question, top_k=5)
    context = "\n".join([d['text'] for d in context_docs])

    # 4. Formate selon le mode
    if mode == "all_months":
        anomalies_str = format_monthly_summary(monthly_results)
    else:
        anomalies_str = format_anomalies(anomalies)

    # 5. Appel LLM
    chain = anomaly_prompt | llm
    response = chain.invoke({
        "anomalies": anomalies_str,
        "period": period,
        "context": context,
        "question": question
    })

    return {
        "agent": "anomaly_agent",
        "mode": mode,
        "period": period,
        "question": question,
        "answer": response.content,
        "anomalies": anomalies,
        "monthly_results": monthly_results,
        "total_transactions_analyzed": len(expenses),
        "anomalies_count": len(anomalies)
    }