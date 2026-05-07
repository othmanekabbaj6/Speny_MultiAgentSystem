# Speny — Système Multi-Agents IA de Gestion Financière

> Projet de contrôle continu — IA Distribuée & Systèmes Multi-Agents  
> Construction d'un Système Multi-Agents avec RAG & Orchestration LangChain

---

## 📋 Table des matières

- [Présentation & Justification](#présentation--justification)
- [Architecture](#architecture)
- [Stack technique](#stack-technique)
- [Structure du projet](#structure-du-projet)
- [Les 10 agents](#les-10-agents)
- [Pipeline RAG](#pipeline-rag)
- [Flux d'orchestration](#flux-dorchestration)
- [Orchestration LangGraph](#orchestration-langgraph)
- [Mémoire](#mémoire)
- [API FastAPI](#api-fastapi)
- [Données de test](#données-de-test)
- [RAG vs Sans RAG](#rag-vs-sans-rag)
- [Installation](#installation)
- [Lancer le projet](#lancer-le-projet)
- [Tests](#tests)
- [Résultats](#résultats)

---

## 🎯 Présentation & Justification

### Le projet

SpendWise est un système multi-agents IA de gestion financière personnelle.
Il analyse les transactions d'un utilisateur, détecte les anomalies,
prédit les dépenses futures, simule des scénarios what-if et génère
des recommandations personnalisées en langage naturel.

**Exemples de questions supportées :**

- _"Est-ce que j'ai dépassé mon budget ce mois-ci ?"_
- _"Combien vais-je dépenser le mois prochain ?"_
- _"Que se passe-t-il si je réduis mes dépenses de 20% ?"_
- _"Donne-moi une analyse complète de mes finances"_

### Justification du cas d'usage

La gestion financière personnelle est un domaine particulièrement adapté
à une architecture multi-agents pour trois raisons :

1. **Multidimensionnalité** — Une analyse financière complète nécessite
   simultanément : détection d'anomalies, prévisions ML, analyse comportementale,
   suivi d'objectifs et simulation de scénarios. Aucun agent unique ne peut
   couvrir efficacement toutes ces dimensions.

2. **Données privées & RAG** — Les transactions bancaires sont des données
   privées non présentes dans les LLMs. Le RAG est indispensable pour ancrer
   les réponses dans les données réelles de l'utilisateur, évitant toute
   hallucination sur les montants ou les dates.

3. **Spécialisation des agents** — Chaque aspect (budget, comportement, prévision,
   simulation) requiert une logique métier distincte. La séparation en agents
   spécialisés permet une maintenance claire et une extensibilité naturelle.

---

## 🏗️ Architecture

```
Question utilisateur
        ↓
┌─────────────────────────┐
│      LangGraph          │  ← Routage conditionnel
│   (graph.py)            │     analyse la question
└─────────────────────────┘     décide quels agents appeler
        ↓
┌──────────────────────────────────────────────────┐
│                  ADVICE AGENT                    │
│            (orchestrateur central)               │
└──────────────────────────────────────────────────┘
        ↓ appelle selon le besoin
┌────────┬────────┬─────────┬───────────┬──────────┬──────────┐
│ Budget │  Goal  │ Anomaly │Behavioral │  Persona │ Forecast │
│ Agent  │ Agent  │  Agent  │   Agent   │  Agent   │  Agent   │
└────────┴────────┴─────────┴───────────┴──────────┴──────────┘
        ↓ enrichi par
┌──────────────────┐    ┌──────────────────┐
│  Retrieval Agent │    │ Simulation Agent │
│  (RAG + ReAct)   │    │  (scénarios)     │
└──────────────────┘    └──────────────────┘
        ↓
┌──────────────────────┐
│  Explanation Agent   │  ← Traduit en langage simple
└──────────────────────┘
        ↓
   Réponse finale
```

---

## 🛠️ Stack technique

| Composant         | Technologie                             |
| ----------------- | --------------------------------------- |
| Langage           | Python 3.10                             |
| LLM               | Groq — llama-3.3-70b-versatile          |
| Orchestration LLM | LangChain 0.3.x + LangGraph             |
| RAG               | LlamaIndex + ChromaDB                   |
| Embeddings        | all-MiniLM-L6-v2 (SentenceTransformers) |
| Prévisions ML     | LSTM (TensorFlow/Keras)                 |
| Base de données   | Firebase Firestore                      |
| API               | FastAPI + Uvicorn                       |
| Pattern agents    | ReAct (Reasoning + Acting)              |

---

## 📁 Structure du projet

```
spendwise/
├── agents/
│   ├── budget_agent.py         # Analyse budgétaire mensuelle
│   ├── goal_agent.py           # Suivi des objectifs financiers
│   ├── anomaly_agent.py        # Détection d'anomalies statistiques
│   ├── behavioral_agent.py     # Analyse des habitudes de dépenses
│   ├── persona_agent.py        # Profil financier (scoring)
│   ├── forecast_agent.py       # Prévisions LSTM
│   ├── simulation_agent.py     # Scénarios what-if
│   ├── retrieval_agent.py      # RAG sémantique + ReAct
│   ├── advice_agent.py         # Orchestrateur central
│   └── explanation_agent.py    # Traduction en langage naturel
├── orchestration/
│   └── graph.py                # Workflow conditionnel LangGraph
├── memory/
│   ├── short_term.py           # Mémoire de session (RAM)
│   └── long_term.py            # Mémoire persistante (Firestore)
├── rag/
│   ├── indexer.py              # Indexation ChromaDB
│   └── query_engine.py         # Recherche sémantique
├── data/
│   └── firebase_client.py      # Client Firestore
├── api/
│   └── main.py                 # API FastAPI
├── config/
│   └── settings.py             # Configuration (.env)
├── tests/
│   └── test_agents.py          # Tests de tous les agents
├── .env.example                # Variables d'environnement
└── requirements.txt            # Dépendances Python
```

---

## 🤖 Les 10 agents

### 1. Budget Agent

**Rôle :** Analyse les dépenses du mois courant et détecte les dépassements de budget.

**Outils :** Requêtes Firestore directes (transactions + budgets du mois)

**Logique :**

- Calcule les dépenses réelles par catégorie
- Compare aux budgets définis par l'utilisateur
- Calcule le taux de consommation (%) et alerte sur les dépassements

**Prompt :** Reçoit un résumé structuré des dépenses vs budgets et génère
une analyse textuelle avec les catégories à risque.

---

### 2. Goal Agent

**Rôle :** Suit la progression vers les objectifs financiers.

**Outils :** Requêtes Firestore (transactions, revenus, objectifs)

**Logique :**

- Calcule l'épargne mensuelle disponible (revenus - dépenses)
- Estime le temps restant pour atteindre chaque objectif
- Identifie les objectifs prioritaires selon la progression

**Prompt :** Reçoit les objectifs et l'épargne disponible, génère un plan
de priorisation en langage naturel.

---

### 3. Anomaly Agent

**Rôle :** Détecte les dépenses inhabituelles par analyse statistique.

**Outils :** Requêtes Firestore + calculs statistiques (moyenne, écart-type)

**Logique :**

- Calcule la moyenne et l'écart-type par marchand sur l'historique
- Identifie les transactions dépassant 2 écarts-types
- Supporte 3 modes : `monthly`, `global`, `all_months`

**Prompt :** Reçoit la liste des anomalies détectées et génère une explication
des transactions suspectes avec leur contexte.

---

### 4. Behavioral Agent

**Rôle :** Analyse les patterns de comportement financier.

**Outils :** Requêtes Firestore (historique complet des transactions)

**Logique :**

- Identifie le jour de la semaine le plus dépensier
- Détermine la catégorie dominante en volume et en fréquence
- Calcule la tendance générale (hausse/baisse en %)
- Analyse la fréquence des dépenses par catégorie

**Prompt :** Reçoit les statistiques comportementales agrégées et génère
des insights sur les habitudes de dépenses de l'utilisateur.

---

### 5. Persona Agent

**Rôle :** Calcule le profil financier de l'utilisateur avec un scoring 0-100.

**Outils :** Agrège les résultats du Budget Agent et du Goal Agent

**Logique :**

- Score épargne (0-40) : taux d'épargne mensuel
- Score discipline budget (0-40) : respect des budgets définis
- Score régularité (0-20) : stabilité des dépenses dans le temps
- Attribution d'un type de profil selon le score total
- Calcul de la tolérance au risque (faible / modérée / élevée)

**Prompt :** Reçoit les scores calculés et génère un portrait financier
personnalisé avec des recommandations adaptées au profil.

---

### 6. Forecast Agent

**Rôle :** Prédit les dépenses du mois prochain avec un modèle LSTM.

**Outils :** Modèle LSTM entraîné sur l'historique Firestore (TensorFlow/Keras)

**Logique :**

- Entraîne un LSTM distinct par catégorie de dépense
- Utilise une fenêtre glissante de 3 mois pour la prédiction
- Calcule la tendance (hausse/baisse en %) vs le mois précédent
- Identifie les catégories à risque de dépassement budgétaire

> Note : LlamaIndex est utilisé pour le chargement du modèle LSTM.
> Le message `"LLM is explicitly disabled. Using MockLLM."` au démarrage
> est normal et n'est pas une erreur.

**Prompt :** Reçoit les prévisions par catégorie et génère une analyse
prospective avec les points de vigilance pour le mois suivant.

---

### 7. Simulation Agent

**Rôle :** Simule des scénarios financiers what-if.

**Outils :** Calculs financiers sur les données Firestore

**3 scénarios disponibles :**

- `category_reduction` : impact d'une réduction de dépenses sur une catégorie
- `income_increase` : impact d'une augmentation de revenus sur l'épargne
- `goal_achievement` : estimation du temps pour atteindre un objectif

**Prompt :** Reçoit les résultats chiffrés de la simulation et génère
une explication de l'impact concret sur la situation financière.

---

### 8. Retrieval Agent _(ReAct)_

**Rôle :** Répond à des questions en langage naturel via RAG + ReAct.

**Pattern ReAct implémenté :**

```
Question → Thought → Action → Observation → Final Answer
```

> ⚠️ Note technique : LangChain 1.2.17 ne fournit pas `create_react_agent`.
> Le pattern ReAct est implémenté manuellement via deux PromptTemplates
> enchaînés (step1 → parse → tool → step2) dans `retrieval_agent.py`.

**3 outils disponibles :**

- `RechercheRAG` : recherche sémantique dans ChromaDB (506 documents)
- `ResumeFinancier` : résumé structuré des 30 derniers jours depuis Firestore
- `BudgetStatus` : statut exact des budgets via Budget Agent

**Prompt step1 :** Analyse la question et décide quel outil utiliser
(Thought + Action).

**Prompt step2 :** Reçoit l'observation de l'outil et génère la réponse finale
en langage naturel.

---

### 9. Advice Agent _(Orchestrateur)_

**Rôle :** Agrège tous les agents et génère une recommandation globale cohérente.

**Outils :** Appelle directement les 6 agents spécialisés

**4 modes disponibles :**

- `full` (~90s) : Budget + Goal + Anomaly + Behavioral + Persona + Forecast
- `quick` (~20s) : Budget + Goal uniquement
- `forecast` (~30s) : Forecast + Persona
- `profile` (~15s) : Persona uniquement

**Prompt :** Reçoit les résumés de chaque agent et génère une synthèse
cohérente avec des recommandations prioritaires classées par impact.

---

### 10. Explanation Agent

**Rôle :** Traduit les outputs techniques en langage naturel simple.

**Outils :** Appelle n'importe quel agent et post-traite sa réponse

**3 audiences supportées :**

- `grand public` : langage simple, analogies du quotidien, emojis
- `expert` : terminologie financière, chiffres détaillés, tendances
- `débutant` : explications pas-à-pas, définitions des concepts

**Prompt :** Reçoit l'output brut d'un agent et le reformule selon
l'audience cible avec la structure et le vocabulaire appropriés.

---

## 🔍 Pipeline RAG

### Vue d'ensemble

```
Firestore (données privées)
        ↓
   indexer.py
   ┌─────────────────────────────────────────┐
   │  1. Récupération des données Firestore  │
   │  2. Conversion en Documents LlamaIndex  │
   │  3. Chunking par transaction            │
   │  4. Génération des embeddings           │
   │     (all-MiniLM-L6-v2)                  │
   │  5. Stockage dans ChromaDB              │
   └─────────────────────────────────────────┘
        ↓
   ChromaDB (506 documents indexés)
        ↓
   query_engine.py
   ┌─────────────────────────────────────────┐
   │  1. Embedding de la question            │
   │  2. Recherche par similarité cosinus    │
   │  3. Récupération top-k documents        │
   │  4. Injection dans le prompt LLM        │
   └─────────────────────────────────────────┘
        ↓
   Retrieval Agent → Réponse contextualisée
```

### Type de données indexées

| Type         | Volume | Champs indexés                                  |
| ------------ | ------ | ----------------------------------------------- |
| Transactions | ~500   | montant, date, marchand, catégorie, description |
| Budgets      | ~5     | catégorie, montant mensuel alloué               |
| Objectifs    | ~4     | nom, montant cible, progression actuelle        |

### Stratégie de chunking

Chaque transaction est indexée comme un **document atomique indépendant**.
Ce choix est justifié par la nature des données :

- Une transaction est une unité sémantique complète (qui, quand, combien, où)
- Le chunking par phrase n'est pas adapté à des données tabulaires structurées
- La granularité fine permet une recherche précise sur un marchand ou une date

**Format d'un document indexé :**

```
Transaction: 450.0 MAD chez Carrefour le 2026-03-15
Catégorie: Food | Marchand: Carrefour
```

### Modèle d'embedding

- **Modèle :** `all-MiniLM-L6-v2` (SentenceTransformers)
- **Dimension :** 384
- **Choix :** Léger, rapide, multilingue, performant sur des textes courts

### Vector Store

- **ChromaDB** en mode persistant local (`./chroma_db`)
- Similarité cosinus pour la recherche
- Top-5 documents retournés par requête

---

## 🔄 Flux d'orchestration

### Comment les agents collaborent

La collaboration entre agents suit un flux en deux niveaux :

**Niveau 1 — LangGraph route la question**

```
Question → Classificateur LLM → Node approprié → Agents ciblés
```

**Niveau 2 — Advice Agent orchestre les agents spécialisés**

```
Advice Agent
    ├── appelle Budget Agent  → résumé budget
    ├── appelle Goal Agent    → résumé objectifs
    ├── appelle Anomaly Agent → résumé anomalies
    ├── appelle Behavioral Agent → résumé comportement
    ├── appelle Persona Agent → profil utilisateur
    ├── appelle Forecast Agent → prévisions
    └── fusionne les 6 résumés → prompt LLM → réponse synthétique
```

**Niveau 3 — Enrichissement contextuel**

```
Retrieval Agent (RAG) ──→ injecte le contexte des transactions
                           dans n'importe quel agent qui le demande
```

**Niveau 4 — Post-traitement**

```
Explanation Agent ──→ reçoit l'output de l'Advice Agent
                       reformule selon l'audience cible
```

### Exemple de scénario complet

```
Utilisateur : "Analyse complète de mes finances"
        ↓
LangGraph → node "full"
        ↓
Advice Agent (mode full)
    ├── Budget Agent    → "Food dépassé à 185%"
    ├── Goal Agent      → "Épargne 4940 MAD/mois"
    ├── Anomaly Agent   → "14 anomalies détectées"
    ├── Behavioral Agent→ "Mardi jour le plus dépensier"
    ├── Persona Agent   → "Score 48/100"
    └── Forecast Agent  → "21730 MAD prévus"
        ↓
LLM synthétise les 6 résumés
        ↓
Explanation Agent → reformule pour le grand public
        ↓
Réponse finale en langage naturel
```

---

## 🕸️ Orchestration LangGraph

LangGraph implémente un workflow conditionnel qui évite d'appeler
tous les agents pour chaque question.

```
"Budget dépassé ?"     → router → budget node    (2 agents)
"Prévisions ?"         → router → forecast node  (2 agents)
"Dépenses bizarres ?"  → router → anomaly node   (1 agent)
"Analyse complète ?"   → router → full node      (6 agents)
```

**Nodes du graph :**

| Node         | Agents appelés                   |
| ------------ | -------------------------------- |
| `budget`     | Budget Agent + Goal Agent        |
| `forecast`   | Forecast Agent + Persona Agent   |
| `anomaly`    | Anomaly Agent                    |
| `behavioral` | Behavioral Agent + Persona Agent |
| `persona`    | Persona Agent                    |
| `full`       | Tous les agents via Advice Agent |
| `aggregator` | Fusion des résultats + LLM final |

---

## 🧠 Mémoire

### Court terme (`memory/short_term.py`)

Stocke l'historique de la conversation en RAM.
Réinitialisée à chaque redémarrage.

```python
session = session_manager.get_session(user_id)
session.add_user_message("Comment vont mes finances ?")
session.add_assistant_message("Analyse en cours...")
history = session.get_history_as_text()
```

### Long terme (`memory/long_term.py`)

Persiste les données dans Firestore.
Survit aux redémarrages.

```python
# Préférences utilisateur
save_preferences(user_id, {"mode": "quick", "audience": "grand public"})

# Historique des conseils
save_advice_to_history(user_id, question, answer, agents_used)

# Notes personnelles
save_user_note(user_id, "Économiser pour un objectif en décembre")
```

---

## 🚀 API FastAPI

L'API expose tous les agents via des endpoints REST.

**Documentation interactive :** http://127.0.0.1:8000/docs

| Méthode | Endpoint                    | Description                      |
| ------- | --------------------------- | -------------------------------- |
| GET     | `/health`                   | Statut de l'API                  |
| POST    | `/api/advice`               | Analyse complète (orchestrateur) |
| POST    | `/api/chat`                 | Question libre (RAG + ReAct)     |
| GET     | `/api/budget/{user_id}`     | Analyse budgétaire               |
| GET     | `/api/goals/{user_id}`      | Objectifs financiers             |
| POST    | `/api/anomalies`            | Détection d'anomalies            |
| GET     | `/api/behavioral/{user_id}` | Habitudes de dépenses            |
| GET     | `/api/persona/{user_id}`    | Profil financier                 |
| GET     | `/api/forecast/{user_id}`   | Prévisions LSTM                  |
| POST    | `/api/simulate`             | Simulation de scénarios          |
| POST    | `/api/explain`              | Explication en langage simple    |
| POST    | `/api/explain/concept`      | Explication d'un concept         |

---

## 🗄️ Données de test

### Utilisateur de test

```
USER_ID   : S6pwTrQB8R7GyuvBdyp0
Currency  : MAD (dirhams marocains)
Période   : mars 2025 → avril 2026
Volume    : ~500 transactions
```

### Données disponibles dans Firestore

| Collection     | Contenu                                                |
| -------------- | ------------------------------------------------------ |
| `transactions` | ~500 transactions (montant, date, marchand, catégorie) |
| `budgets`      | Food (4000 MAD/mois), Education (552 MAD/mois)         |
| `goals`        | Trip Japan (40%), New Laptop (60%), New House (0%)     |

### Réindexer les données dans ChromaDB

Si ChromaDB est vide ou doit être reconstruit, lancer :

```bash
python -m rag.indexer
```

Cela récupère toutes les données Firestore et reconstruit l'index vectoriel
(~506 documents, durée ~30 secondes).

---

## ⚖️ RAG vs Sans RAG

Le RAG est central dans SpendWise. Voici la différence concrète :

### Sans RAG — Réponse générique et incorrecte

```
Question : "Est-ce que j'ai dépassé mon budget ce mois-ci ?"

Réponse LLM (sans RAG) :
"Il est difficile de répondre sans connaître vos données financières.
En général, un budget est dépassé lorsque les dépenses excèdent les
revenus alloués. Je vous recommande de vérifier votre application bancaire."
```

→ Réponse inutile, aucune donnée réelle, aucune valeur ajoutée.

### Avec RAG — Réponse précise et contextualisée

```
Question : "Est-ce que j'ai dépassé mon budget ce mois-ci ?"

Contexte injecté par RAG (extrait ChromaDB) :
- Budget Food : 4000 MAD/mois
- Dépenses Food avril 2026 : 7431 MAD
- Transactions : Carrefour 450 MAD, Restaurant X 320 MAD...

Réponse LLM (avec RAG) :
"Oui, votre budget Food est dépassé à 185.8% en avril 2026.
Vous avez dépensé 7431 MAD pour un budget de 4000 MAD, soit
un dépassement de 3431 MAD. Les principales dépenses proviennent
de Carrefour et des restaurants. Je recommande de réduire les
sorties au restaurant pour les prochaines semaines."
```

→ Réponse précise, chiffrée, actionnable, ancrée dans les vraies données.

---

## 🎁 Livrables inclus

Ce dépôt contient les éléments attendus pour le livrable :

- Code source complet et fonctionnel organisé par agents.
- `README.md` détaillé avec architecture, installation et exécution.
- `requirements.txt` listant toutes les dépendances Python.
- `.env.example` template pour les clés API sans valeurs sensibles.
- Documentation technique intégrée dans le README.
- Démonstration possible via l'API FastAPI et les tests de `tests/test_agents.py`.

## ⚙️ Installation

### 1. Cloner le projet

```bash
git clone
cd spendwise
```

### 2. Créer l'environnement virtuel

```bash
python -m venv .venv
.venv\Scripts\activate      # Windows
source .venv/bin/activate   # Linux/Mac
```

### 3. Installer les dépendances

```bash
pip install -r requirements.txt
```

### 4. Configurer les variables d'environnement

```bash
cp .env.example .env
# Remplir .env avec tes clés
```

Le fichier `.env` doit contenir :

```bash
GROQ_API_KEY=your_groq_key_here
LLM_MODEL=llama-3.3-70b-versatile
FIREBASE_PROJECT_ID=your_firebase_project_id_here
FIREBASE_CREDENTIALS_PATH=firebase_credentials.json
CHROMA_PERSIST_DIR=.chroma_db
```

### 5. Ajouter les credentials Firebase

Télécharge le fichier `firebase_credentials.json` depuis la console Firebase
et place-le à la racine du projet.

### 6. Indexer les données dans ChromaDB

```bash
python -m rag.indexer
```

---

## ▶️ Lancer le projet

### Lancer l'API

```bash
uvicorn api.main:app --reload --port 8000
```

### Accéder à la documentation interactive

```
http://127.0.0.1:8000/docs
```

---

## 🧪 Tests

Tester un agent spécifique :

```bash
python -m tests.test_agents budget
python -m tests.test_agents goal
python -m tests.test_agents anomaly
python -m tests.test_agents behavioral
python -m tests.test_agents persona
python -m tests.test_agents forecast
python -m tests.test_agents simulation
python -m tests.test_agents retrieval
python -m tests.test_agents retrieval_react
python -m tests.test_agents advice
python -m tests.test_agents explanation
python -m tests.test_agents graph
python -m tests.test_agents memory
```

Tester tous les agents :

```bash
python -m tests.test_agents all
```

---

## 📊 Résultats

Résultats obtenus sur l'utilisateur de test (`USER_ID = S6pwTrQB8R7GyuvBdyp0`) :

| Agent            | Résultat                                                    |
| ---------------- | ----------------------------------------------------------- |
| Budget Agent     | Food dépassé à **185.8%** (7431 MAD / 4000 MAD)             |
| Goal Agent       | Épargne mensuelle **4940 MAD** (taux 23.4%)                 |
| Anomaly Agent    | **14 anomalies** détectées sur l'historique                 |
| Behavioral Agent | Mardi = jour le plus dépensier, Food = **31%** des dépenses |
| Persona Agent    | Profil **"en développement"**, score **48/100**             |
| Forecast Agent   | **21 730 MAD** prévus pour le mois prochain (LSTM)          |
| Simulation Agent | Réduction Food -20% → économie **1 533 MAD/mois**           |
| Retrieval Agent  | RAG sémantique + ReAct sur **506 documents**                |
| LangGraph        | Routage conditionnel — **5 routes** possibles               |
| Mémoire          | Short-term (RAM) + Long-term (Firestore) ✅                 |
| API FastAPI      | 12 endpoints REST opérationnels ✅                          |

---

## 👥 Auteurs

Projet réalisé dans le cadre du cours **IA Distribuée & Systèmes Multi-Agents**.
