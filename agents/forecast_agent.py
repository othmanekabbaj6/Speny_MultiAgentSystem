import pandas as pd
import numpy as np
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from data.firebase_client import get_transactions, get_budgets
from rag.query_engine import search_user_data
from config.settings import settings
from datetime import datetime, timezone
import logging
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

llm = ChatGroq(
    api_key=settings.groq_api_key,
    model_name=settings.llm_model
)

forecast_prompt = PromptTemplate(
    input_variables=["forecasts", "budgets", "context", "question"],
    template="""
Tu es un analyste financier expert en prévision de dépenses.
Analyse les prévisions suivantes et réponds en français.

PRÉVISIONS DES DÉPENSES DU MOIS PROCHAIN :
{forecasts}

BUDGETS DÉFINIS :
{budgets}

CONTEXTE SUPPLÉMENTAIRE :
{context}

QUESTION : {question}

Pour chaque catégorie prévue :
- Indique si la dépense prévue est dans le budget ou non
- Compare avec les dépenses habituelles
- Signale les catégories à risque de dépassement
- Propose des actions préventives concrètes

Termine par un résumé du budget total prévu pour le mois prochain.
Sois précis et actionnable.
"""
)

# ─── Préparation des données mensuelles ───────────────────────
def prepare_monthly_data(
    transactions: list[dict],
    category: str
) -> pd.DataFrame:
    """
    Agrège les transactions par mois pour une catégorie.
    Nettoie les spikes > 2.5x la médiane.
    """
    cat_txs = [
        t for t in transactions
        if t.get('category', '').lower() == category.lower()
        and t.get('type') == 'expense'
    ]

    if not cat_txs:
        return pd.DataFrame()

    monthly = {}
    for t in cat_txs:
        date = t.get('date')
        if hasattr(date, 'tzinfo') and date.tzinfo is None:
            date = date.replace(tzinfo=timezone.utc)
        if date:
            key = f"{date.year}-{date.month:02d}-01"
            monthly[key] = monthly.get(key, 0) + t.get('amount', 0)

    if len(monthly) < 3:
        return pd.DataFrame()

    df = pd.DataFrame([
        {'ds': pd.Timestamp(k), 'y': round(v, 2)}
        for k, v in sorted(monthly.items())
    ])

    # Nettoyage des spikes > 2.5x la médiane
    median = df['y'].median()
    spike_threshold = median * 2.5
    n_spikes = (df['y'] > spike_threshold).sum()
    if n_spikes > 0:
        logger.info(
            f"{category} : {n_spikes} spike(s) remplacé(s) "
            f"(seuil: {spike_threshold:.0f} MAD)"
        )
        df['y'] = df['y'].apply(
            lambda v: median if v > spike_threshold else v
        )

    return df

# ─── Normalisation ────────────────────────────────────────────
def normalize(data: np.ndarray) -> tuple:
    """Min-Max normalization → [0, 1]"""
    min_val = data.min()
    max_val = data.max()
    if max_val == min_val:
        return np.zeros_like(data), min_val, max_val
    normalized = (data - min_val) / (max_val - min_val)
    return normalized, min_val, max_val

def denormalize(value: float, min_val: float, max_val: float) -> float:
    """Inverse normalization"""
    if max_val == min_val:
        return min_val
    return value * (max_val - min_val) + min_val

# ─── Construction du dataset LSTM ─────────────────────────────
def create_sequences(data: np.ndarray, window: int = 3) -> tuple:
    """
    Crée des séquences pour LSTM.
    window = nombre de mois passés utilisés pour prédire le suivant.
    """
    X, y = [], []
    for i in range(len(data) - window):
        X.append(data[i:i + window])
        y.append(data[i + window])
    return np.array(X), np.array(y)

# ─── Modèle LSTM ──────────────────────────────────────────────
def build_lstm_model(window: int = 3):
    """
    Construit un modèle LSTM simple.
    Architecture légère adaptée aux petits datasets financiers.
    """
    try:
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import LSTM, Dense, Dropout
        from tensorflow.keras.optimizers import Adam
    except ImportError:
        raise ImportError("tensorflow non installé. Lance : pip install tensorflow")

    model = Sequential([
        LSTM(
            32,
            input_shape=(window, 1),
            return_sequences=False
        ),
        Dropout(0.1),
        Dense(16, activation='relu'),
        Dense(1)
    ])

    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='mse'
    )

    return model

# ─── Prévision LSTM pour une catégorie ───────────────────────
def forecast_category_lstm(
    transactions: list[dict],
    category: str,
    window: int = 3
) -> dict:
    """
    Prédit les dépenses du mois prochain avec LSTM.
    Fallback sur moyenne pondérée si pas assez de données.
    """
    df = prepare_monthly_data(transactions, category)

    if df.empty:
        return None

    values = df['y'].values.astype(float)
    historical_mean = round(float(np.mean(values)), 2)
    historical_std = round(float(np.std(values)), 2)
    last_month = round(float(values[-1]), 2)

    # Fallback si pas assez de données pour LSTM
    if len(values) < window + 2:
        logger.info(f"{category} : données insuffisantes → moyenne pondérée")
        return _weighted_mean_forecast(
            category, values, historical_mean,
            historical_std, last_month, len(df)
        )

    try:
        # Normalisation
        norm_values, min_val, max_val = normalize(values)

        # Séquences
        X, y_train = create_sequences(norm_values, window)
        X = X.reshape(X.shape[0], X.shape[1], 1)

        # Modèle LSTM
        model = build_lstm_model(window)

        # Entraînement silencieux
        model.fit(
            X, y_train,
            epochs=200,
            batch_size=max(1, len(X) // 2),
            verbose=0
        )

        # Prédiction : utilise les `window` derniers mois
        last_sequence = norm_values[-window:].reshape(1, window, 1)
        predicted_norm = float(model.predict(last_sequence, verbose=0)[0][0])

        # Dénormalisation
        predicted_raw = denormalize(predicted_norm, min_val, max_val)

        # Sanity check : entre 40% et 220% de la moyenne historique
        min_acceptable = historical_mean * 0.4
        max_acceptable = historical_mean * 2.2

        if min_acceptable <= predicted_raw <= max_acceptable:
            predicted = round(predicted_raw, 2)
            method = 'lstm'
        else:
            logger.warning(
                f"{category} : LSTM hors limites "
                f"({predicted_raw:.0f} MAD) → moyenne pondérée"
            )
            return _weighted_mean_forecast(
                category, values, historical_mean,
                historical_std, last_month, len(df)
            )

        # Intervalle de confiance basé sur l'écart-type historique
        lower = round(max(0, predicted - historical_std * 0.5), 2)
        upper = round(predicted + historical_std * 0.5, 2)

        trend_pct = round(
            (predicted - historical_mean) / historical_mean * 100, 1
        ) if historical_mean > 0 else 0

        logger.info(f"{category} → {predicted} MAD (LSTM)")

        return {
            'category': category,
            'predicted': predicted,
            'lower_bound': lower,
            'upper_bound': upper,
            'historical_mean': historical_mean,
            'historical_std': historical_std,
            'last_month': last_month,
            'trend_pct': trend_pct,
            'data_points': len(df),
            'method': method
        }

    except Exception as e:
        logger.error(f"Erreur LSTM pour {category}: {e}")
        return _weighted_mean_forecast(
            category, values, historical_mean,
            historical_std, last_month, len(df)
        )

# ─── Fallback : Moyenne pondérée ──────────────────────────────
def _weighted_mean_forecast(
    category: str,
    values: np.ndarray,
    historical_mean: float,
    historical_std: float,
    last_month: float,
    data_points: int
) -> dict:
    """
    Moyenne pondérée : derniers mois ont plus de poids.
    Utilisée quand LSTM n'est pas applicable.
    """
    n = len(values)
    weights = np.ones(n)
    weights[-min(3, n):] = 3.0
    weights = weights / weights.sum()
    predicted = round(float(np.average(values, weights=weights)), 2)

    trend_pct = round(
        (predicted - historical_mean) / historical_mean * 100, 1
    ) if historical_mean > 0 else 0

    logger.info(f"{category} → {predicted} MAD (moyenne pondérée)")

    return {
        'category': category,
        'predicted': predicted,
        'lower_bound': round(max(0, predicted - historical_std * 0.5), 2),
        'upper_bound': round(predicted + historical_std * 0.5, 2),
        'historical_mean': historical_mean,
        'historical_std': historical_std,
        'last_month': last_month,
        'trend_pct': trend_pct,
        'data_points': data_points,
        'method': 'weighted_mean'
    }

# ─── Prévision de toutes les catégories ──────────────────────
def forecast_all_categories(transactions: list[dict]) -> list[dict]:
    """
    Lance la prévision LSTM pour toutes les catégories.
    """
    expenses = [t for t in transactions if t.get('type') == 'expense']
    categories = list(set(
        t.get('category', '') for t in expenses
        if t.get('category')
    ))

    forecasts = []
    for cat in categories:
        result = forecast_category_lstm(transactions, cat)
        if result:
            forecasts.append(result)

    forecasts.sort(key=lambda x: x['predicted'], reverse=True)
    return forecasts

# ─── Formatage ────────────────────────────────────────────────
def format_forecasts(forecasts: list[dict], budgets: list[dict]) -> str:
    budget_map = {
        b.get('category', '').lower(): b.get('limit_amount', 0)
        for b in budgets
    }

    lines = []
    total_predicted = sum(f['predicted'] for f in forecasts)

    for f in forecasts:
        cat = f['category']
        predicted = f['predicted']
        method = f.get('method', '?')
        budget_limit = budget_map.get(cat.lower())

        if budget_limit:
            ratio = predicted / budget_limit * 100
            if ratio > 100:
                budget_status = f"🔴 Dépassement prévu ({ratio:.0f}% du budget {budget_limit} MAD)"
            elif ratio > 80:
                budget_status = f"🟠 Proche de la limite ({ratio:.0f}% du budget {budget_limit} MAD)"
            else:
                budget_status = f"🟢 Dans le budget ({ratio:.0f}% du budget {budget_limit} MAD)"
        else:
            budget_status = "⚪ Pas de budget défini"

        trend = f"↑ +{f['trend_pct']}%" if f['trend_pct'] > 0 else \
                f"↓ {f['trend_pct']}%" if f['trend_pct'] < 0 else "→ stable"

        lines.append(
            f"- {cat} : {predicted} MAD "
            f"(intervalle: {f['lower_bound']}–{f['upper_bound']} MAD) | "
            f"Moyenne historique: {f['historical_mean']} MAD | "
            f"Tendance: {trend} | {budget_status} | "
            f"Méthode: {method}"
        )

    lines.append(f"\nTOTAL PRÉVU : {round(total_predicted, 2)} MAD")
    return "\n".join(lines)

# ─── Agent principal ───────────────────────────────────────────
def run_forecast_agent(
    user_id: str,
    question: str = None,
    periods: int = 1
) -> dict:
    """
    Prédit les dépenses du mois prochain par catégorie via LSTM.
    """
    if question is None:
        question = "Quelles seront mes dépenses le mois prochain ? Y a-t-il des risques de dépassement ?"

    transactions = get_transactions(user_id, limit=500)
    budgets = get_budgets(user_id)

    logger.info("Lancement des prévisions LSTM...")
    forecasts = forecast_all_categories(transactions)

    context_docs = search_user_data(user_id, question, top_k=5)
    context = "\n".join([d['text'] for d in context_docs])

    budgets_str = "\n".join([
        f"- {b.get('category')} : {b.get('limit_amount')} MAD/{b.get('period')}"
        for b in budgets
    ]) if budgets else "Aucun budget défini."

    forecasts_str = format_forecasts(forecasts, budgets)

    chain = forecast_prompt | llm
    response = chain.invoke({
        "forecasts": forecasts_str,
        "budgets": budgets_str,
        "context": context,
        "question": question
    })

    total_predicted = round(sum(f['predicted'] for f in forecasts), 2)
    at_risk = [
        f for f in forecasts
        if any(
            b.get('category', '').lower() == f['category'].lower()
            and f['predicted'] > b.get('limit_amount', float('inf'))
            for b in budgets
        )
    ]

    return {
        "agent": "forecast_agent",
        "question": question,
        "answer": response.content,
        "forecasts": forecasts,
        "total_predicted": total_predicted,
        "at_risk_categories": at_risk,
        "budgets_count": len(budgets)
    }