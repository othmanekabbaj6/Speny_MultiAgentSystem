"""
memory/short_term.py
────────────────────
Mémoire à court terme — stocke l'historique de la session en RAM.
Réinitialisée à chaque redémarrage du serveur.

Utilisée pour :
- Garder le contexte de la conversation en cours
- Permettre des questions de suivi ("et si je réduis de 30% ?")
- Éviter de rappeler les agents si la question est similaire
"""

from datetime import datetime, timezone
from typing import List, Optional
import logging

logger = logging.getLogger(__name__)


# ─── Structure d'un message ───────────────────────────────────

class Message:
    def __init__(self, role: str, content: str, metadata: dict = None):
        self.role      = role       # "user" | "assistant"
        self.content   = content
        self.metadata  = metadata or {}
        self.timestamp = datetime.now(timezone.utc).isoformat()

    def to_dict(self) -> dict:
        return {
            "role":      self.role,
            "content":   self.content,
            "metadata":  self.metadata,
            "timestamp": self.timestamp,
        }


# ─── Session utilisateur ──────────────────────────────────────

class ShortTermMemory:
    """
    Mémoire de session pour un utilisateur.
    Stocke les derniers messages de la conversation.
    """

    def __init__(self, user_id: str, max_messages: int = 20):
        self.user_id      = user_id
        self.max_messages = max_messages
        self.messages: List[Message] = []
        self.context: dict = {}     # contexte libre (résultats agents, etc.)

    def add_user_message(self, content: str, metadata: dict = None):
        """Ajoute un message utilisateur."""
        self.messages.append(Message("user", content, metadata))
        self._trim()

    def add_assistant_message(self, content: str, metadata: dict = None):
        """Ajoute une réponse de l'assistant."""
        self.messages.append(Message("assistant", content, metadata))
        self._trim()

    def get_history(self, last_n: int = 10) -> List[dict]:
        """Retourne les derniers N messages."""
        return [m.to_dict() for m in self.messages[-last_n:]]

    def get_history_as_text(self, last_n: int = 6) -> str:
        """Retourne l'historique formaté en texte pour le LLM."""
        recent = self.messages[-last_n:]
        lines = []
        for m in recent:
            prefix = "Utilisateur" if m.role == "user" else "Assistant"
            lines.append(f"{prefix}: {m.content[:300]}")
        return "\n".join(lines)

    def set_context(self, key: str, value):
        """Stocke une valeur dans le contexte de session."""
        self.context[key] = value

    def get_context(self, key: str, default=None):
        """Récupère une valeur du contexte."""
        return self.context.get(key, default)

    def clear(self):
        """Efface la mémoire de session."""
        self.messages = []
        self.context  = {}
        logger.info(f"Session cleared for user {self.user_id}")

    def _trim(self):
        """Garde uniquement les N derniers messages."""
        if len(self.messages) > self.max_messages:
            self.messages = self.messages[-self.max_messages:]

    def summary(self) -> dict:
        return {
            "user_id":       self.user_id,
            "message_count": len(self.messages),
            "context_keys":  list(self.context.keys()),
        }


# ─── Gestionnaire global des sessions ─────────────────────────

class SessionManager:
    """
    Gère toutes les sessions actives en mémoire.
    Une session par user_id.
    """

    def __init__(self):
        self._sessions: dict[str, ShortTermMemory] = {}

    def get_session(self, user_id: str) -> ShortTermMemory:
        """Retourne la session existante ou en crée une nouvelle."""
        if user_id not in self._sessions:
            self._sessions[user_id] = ShortTermMemory(user_id)
            logger.info(f"New session created for user {user_id}")
        return self._sessions[user_id]

    def clear_session(self, user_id: str):
        """Efface la session d'un utilisateur."""
        if user_id in self._sessions:
            del self._sessions[user_id]
            logger.info(f"Session deleted for user {user_id}")

    def active_sessions(self) -> List[str]:
        """Retourne la liste des user_ids avec une session active."""
        return list(self._sessions.keys())


# ─── Instance globale ─────────────────────────────────────────
session_manager = SessionManager()