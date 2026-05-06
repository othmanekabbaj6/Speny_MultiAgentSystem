from llama_index.core import VectorStoreIndex
from rag.indexer import build_user_index, load_user_index
from llama_index.core.retrievers import VectorIndexRetriever
import chromadb
from config.settings import settings
import logging

logger = logging.getLogger(__name__)

def get_retriever(user_id: str, top_k: int = 10):
    """
    Retourne un retriever pour chercher
    les documents les plus pertinents.
    """
    try:
        index = load_user_index(user_id)
        logger.info("Index chargé depuis ChromaDB")
    except Exception:
        logger.info("Index non trouvé, création en cours...")
        index = build_user_index(user_id)

    return VectorIndexRetriever(index=index, similarity_top_k=top_k)

def search_user_data(user_id: str, query: str, top_k: int = 10) -> list[dict]:
    """
    Recherche sémantique sur les données d'un user.
    Retourne les documents les plus pertinents.
    """
    retriever = get_retriever(user_id, top_k)
    nodes = retriever.retrieve(query)

    results = []
    for node in nodes:
        results.append({
            "text": node.text,
            "score": round(node.score, 4) if node.score else 0,
            "metadata": node.metadata
        })
    return results  