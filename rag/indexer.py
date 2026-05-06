import chromadb
from llama_index.core import VectorStoreIndex, StorageContext, Document
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings as LlamaSettings
from data.firebase_client import get_transactions, get_budgets, get_goals
from config.settings import settings
import logging

logger = logging.getLogger(__name__)

# ─── Embedding model (local, gratuit) ─────────────────────────
LlamaSettings.embed_model = HuggingFaceEmbedding(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
LlamaSettings.llm = None  # on gère le LLM via LangChain

# ─── ChromaDB client ──────────────────────────────────────────
def get_chroma_collection(collection_name: str):
    chroma_client = chromadb.PersistentClient(path=settings.chroma_persist_dir)
    return chroma_client.get_or_create_collection(collection_name)

# ─── Convertir une transaction en Document LlamaIndex ─────────
def transaction_to_document(t: dict) -> Document:
    date_str = t['date'].strftime('%Y-%m-%d') if hasattr(t['date'], 'strftime') else str(t['date'])
    text = (
        f"Transaction du {date_str} : "
        f"{t.get('description', '')} chez {t.get('merchant_name', 'inconnu')}. "
        f"Catégorie : {t.get('category', '')}. "
        f"Montant : {t.get('amount', 0)} {t.get('currency', 'MAD')}. "
        f"Type : {t.get('type', '')}."
    )
    return Document(
        text=text,
        metadata={
            "id": t.get("id", ""),
            "date": date_str,
            "category": t.get("category", ""),
            "amount": t.get("amount", 0),
            "currency": t.get("currency", "MAD"),
            "type": t.get("type", ""),
            "merchant_name": t.get("merchant_name", ""),
            "user_id": t.get("user_id", ""),
        }
    )

# ─── Convertir un budget en Document ──────────────────────────
def budget_to_document(b: dict) -> Document:
    text = (
        f"Budget {b.get('category', '')} : "
        f"limite de {b.get('limit_amount', 0)} {b.get('currency', 'MAD')} "
        f"par {b.get('period', 'mois')}. "
        f"Alerte à {int(b.get('alert_at', 0.8) * 100)}% du budget."
    )
    return Document(
        text=text,
        metadata={
            "id": b.get("id", ""),
            "category": b.get("category", ""),
            "limit_amount": b.get("limit_amount", 0),
            "period": b.get("period", "monthly"),
            "type": "budget",
            "user_id": b.get("user_id", ""),
        }
    )

# ─── Convertir un goal en Document ────────────────────────────
def goal_to_document(g: dict) -> Document:
    text = (
        f"Objectif financier : {g.get('title', '')}. "
        f"Montant cible : {g.get('target_amount', 0)} MAD. "
        f"Montant actuel : {g.get('current_amount', 0)} MAD. "
        f"Progression : {g.get('progress_pct', 0)}%. "
        f"Statut : {g.get('status', 'active')}."
    )
    return Document(
        text=text,
        metadata={
            "id": g.get("id", ""),
            "title": g.get("title", ""),
            "target_amount": g.get("target_amount", 0),
            "current_amount": g.get("current_amount", 0),
            "progress_pct": g.get("progress_pct", 0),
            "status": g.get("status", "active"),
            "type": "goal",
            "user_id": g.get("user_id", ""),
        }
    )

# ─── Indexation principale ────────────────────────────────────
def build_user_index(user_id: str) -> VectorStoreIndex:
    logger.info(f"Indexation des données pour user: {user_id}")

    # Récupère toutes les données
    transactions = get_transactions(user_id, limit=500)
    budgets = get_budgets(user_id)
    goals = get_goals(user_id)

    # Convertit en documents
    documents = []
    documents += [transaction_to_document(t) for t in transactions]
    documents += [budget_to_document(b) for b in budgets]
    documents += [goal_to_document(g) for g in goals]

    logger.info(f"{len(documents)} documents à indexer")

    # ChromaDB collection par user
    collection = get_chroma_collection(f"spendwise_{user_id}")
    vector_store = ChromaVectorStore(chroma_collection=collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    # Création de l'index
    index = VectorStoreIndex.from_documents(
        documents,
        storage_context=storage_context,
        show_progress=True
    )

    logger.info("Index créé avec succès")
    return index

# ─── Chargement d'un index existant ──────────────────────────
def load_user_index(user_id: str) -> VectorStoreIndex:
    collection = get_chroma_collection(f"spendwise_{user_id}")
    vector_store = ChromaVectorStore(chroma_collection=collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    return VectorStoreIndex.from_vector_store(
        vector_store,
        storage_context=storage_context
    )