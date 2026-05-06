import firebase_admin
from firebase_admin import credentials, firestore
from config.settings import settings
from typing import Optional
import logging

logger = logging.getLogger(__name__)

# Initialisation unique Firebase
def init_firebase():
    if not firebase_admin._apps:
        cred = credentials.Certificate(settings.firebase_credentials_path)
        firebase_admin.initialize_app(cred, {
            'projectId': settings.firebase_project_id
        })
    return firestore.client()

db = init_firebase()

# ─── Helpers centimes ─────────────────────────────────────────
def centimes_to_amount(centimes: int, decimals: int = 2) -> float:
    return round(centimes / 100, decimals)

def amount_to_centimes(amount: float) -> int:
    return int(round(amount * 100))

# ─── USERS ────────────────────────────────────────────────────
def get_user(user_id: str) -> Optional[dict]:
    doc = db.collection('users').document(user_id).get()
    if doc.exists:
        data = doc.to_dict()
        data['id'] = doc.id
        return data
    return None

# ─── TRANSACTIONS ─────────────────────────────────────────────
def get_transactions(user_id: str, limit: int = 100) -> list[dict]:
    docs = (
        db.collection('transactions')
        .where('user_id', '==', user_id)
        .order_by('date', direction=firestore.Query.DESCENDING)
        .limit(limit)
        .stream()
    )
    results = []
    for doc in docs:
        data = doc.to_dict()
        data['id'] = doc.id
        data['amount'] = centimes_to_amount(data.get('amount_centimes', 0))
        results.append(data)
    return results

def get_transactions_by_category(user_id: str, category: str) -> list[dict]:
    docs = (
        db.collection('transactions')
        .where('user_id', '==', user_id)
        .where('category', '==', category)
        .stream()
    )
    results = []
    for doc in docs:
        data = doc.to_dict()
        data['id'] = doc.id
        data['amount'] = centimes_to_amount(data.get('amount_centimes', 0))
        results.append(data)
    return results

# ─── BUDGETS ──────────────────────────────────────────────────
def get_budgets(user_id: str) -> list[dict]:
    docs = (
        db.collection('budgets')
        .where('user_id', '==', user_id)
        .stream()
    )
    results = []
    for doc in docs:
        data = doc.to_dict()
        data['id'] = doc.id
        data['limit_amount'] = centimes_to_amount(data.get('limit_amount_centimes', 0))
        results.append(data)
    return results

# ─── GOALS ────────────────────────────────────────────────────
def get_goals(user_id: str) -> list[dict]:
    docs = (
        db.collection('goals')
        .where('user_id', '==', user_id)
        .stream()
    )
    results = []
    for doc in docs:
        data = doc.to_dict()
        data['id'] = doc.id
        data['target_amount'] = centimes_to_amount(data.get('target_amount_centimes', 0))
        data['current_amount'] = centimes_to_amount(data.get('current_amount_centimes', 0))
        data['progress_pct'] = round(
            data['current_amount'] / data['target_amount'] * 100, 1
        ) if data['target_amount'] > 0 else 0
        results.append(data)
    return results

# ─── CATEGORIES ───────────────────────────────────────────────
def get_categories(user_id: str) -> list[dict]:
    docs = (
        db.collection('categories')
        .where('user_id', '==', user_id)
        .stream()
    )
    results = []
    for doc in docs:
        data = doc.to_dict()
        data['id'] = doc.id
        results.append(data)
    return results