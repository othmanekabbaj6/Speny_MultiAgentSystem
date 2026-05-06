"""
memory/long_term.py
───────────────────
Mémoire à long terme — persiste les préférences utilisateur dans Firestore.
Survit aux redémarrages du serveur.

Stocke :
- Préférences de l'utilisateur (langue, audience, mode favori)
- Historique des conseils reçus (dernières recommandations)
- Résumés des analyses passées
- Objectifs notés par l'utilisateur
"""

from data.firebase_client import db
from datetime import datetime, timezone
from typing import Optional
import logging

logger = logging.getLogger(__name__)

COLLECTION = "user_memory"


# ─── Lecture ──────────────────────────────────────────────────

def get_user_memory(user_id: str) -> dict:
    """
    Récupère la mémoire persistante d'un utilisateur depuis Firestore.
    Retourne un dict vide si aucune mémoire n'existe.
    """
    try:
        doc = db.collection(COLLECTION).document(user_id).get()
        if doc.exists:
            return doc.to_dict()
        return {}
    except Exception as e:
        logger.error(f"get_user_memory error ({user_id}): {e}")
        return {}


# ─── Écriture ─────────────────────────────────────────────────

def save_user_memory(user_id: str, data: dict):
    """
    Sauvegarde ou met à jour la mémoire d'un utilisateur dans Firestore.
    Merge avec les données existantes (ne supprime pas les champs absents).
    """
    try:
        data["updated_at"] = datetime.now(timezone.utc).isoformat()
        db.collection(COLLECTION).document(user_id).set(data, merge=True)
        logger.info(f"Memory saved for user {user_id}")
    except Exception as e:
        logger.error(f"save_user_memory error ({user_id}): {e}")


# ─── Préférences ──────────────────────────────────────────────

def get_preferences(user_id: str) -> dict:
    """
    Retourne les préférences de l'utilisateur.
    Valeurs par défaut si non définies.
    """
    memory = get_user_memory(user_id)
    return memory.get("preferences", {
        "language":  "fr",
        "audience":  "grand public",    # grand public | expert | débutant
        "mode":      "quick",           # full | quick | forecast | profile
        "currency":  "MAD",
    })

def save_preferences(user_id: str, preferences: dict):
    """Met à jour les préférences de l'utilisateur."""
    save_user_memory(user_id, {"preferences": preferences})


# ─── Historique des conseils ──────────────────────────────────

def save_advice_to_history(user_id: str, question: str, answer: str, agents_used: list):
    """
    Sauvegarde un conseil dans l'historique Firestore.
    Garde les 10 derniers conseils maximum.
    """
    try:
        memory   = get_user_memory(user_id)
        history  = memory.get("advice_history", [])

        entry = {
            "question":    question,
            "answer":      answer[:500],    # tronqué pour économiser l'espace
            "agents_used": agents_used,
            "timestamp":   datetime.now(timezone.utc).isoformat(),
        }

        history.append(entry)

        # Garde les 10 derniers
        if len(history) > 10:
            history = history[-10:]

        save_user_memory(user_id, {"advice_history": history})

    except Exception as e:
        logger.error(f"save_advice_to_history error ({user_id}): {e}")


def get_advice_history(user_id: str, last_n: int = 5) -> list:
    """Retourne les N derniers conseils reçus."""
    memory  = get_user_memory(user_id)
    history = memory.get("advice_history", [])
    return history[-last_n:]


# ─── Résumés des analyses ─────────────────────────────────────

def save_analysis_summary(user_id: str, month: int, year: int, summary: dict):
    """
    Sauvegarde le résumé d'une analyse mensuelle.
    Ex: budget dépassé, épargne, score persona.
    """
    try:
        key = f"{year}_{month:02d}"
        save_user_memory(user_id, {f"analysis_{key}": summary})
    except Exception as e:
        logger.error(f"save_analysis_summary error ({user_id}): {e}")


def get_analysis_summary(user_id: str, month: int, year: int) -> dict:
    """Récupère le résumé d'une analyse mensuelle."""
    memory = get_user_memory(user_id)
    key    = f"{year}_{month:02d}"
    return memory.get(f"analysis_{key}", {})


# ─── Notes utilisateur ────────────────────────────────────────

def save_user_note(user_id: str, note: str):
    """
    Sauvegarde une note personnelle de l'utilisateur.
    Ex: "Je veux économiser pour une voiture en décembre"
    """
    try:
        memory = get_user_memory(user_id)
        notes  = memory.get("notes", [])
        notes.append({
            "note":      note,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })
        if len(notes) > 20:
            notes = notes[-20:]
        save_user_memory(user_id, {"notes": notes})
    except Exception as e:
        logger.error(f"save_user_note error ({user_id}): {e}")


def get_user_notes(user_id: str) -> list:
    """Retourne toutes les notes de l'utilisateur."""
    memory = get_user_memory(user_id)
    return memory.get("notes", [])


# ─── Contexte complet pour le LLM ─────────────────────────────

def get_memory_context_for_llm(user_id: str) -> str:
    """
    Retourne un résumé de la mémoire long terme formaté
    pour être injecté dans un prompt LLM.
    """
    lines = []

    prefs = get_preferences(user_id)
    lines.append(f"Préférences : langue={prefs.get('language')}, "
                 f"mode={prefs.get('mode')}, audience={prefs.get('audience')}")

    history = get_advice_history(user_id, last_n=3)
    if history:
        lines.append(f"Derniers conseils ({len(history)}) :")
        for h in history:
            lines.append(f"  - [{h['timestamp'][:10]}] {h['question'][:80]}")

    notes = get_user_notes(user_id)
    if notes:
        lines.append(f"Notes personnelles :")
        for n in notes[-3:]:
            lines.append(f"  - {n['note']}")

    return "\n".join(lines) if lines else "Aucune mémoire long terme disponible."