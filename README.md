# Speny — AI Multi-Agent Personal Finance Management System

> Continuous assessment project — Distributed AI & Multi-Agent Systems
> Building a Multi-Agent System with RAG & LangChain Orchestration

---

## 📋 Table of Contents

- [Overview & Rationale](#overview--rationale)
- [Architecture](#architecture)
- [Tech Stack](#tech-stack)
- [Design Decisions](#design-decisions)
- [Project Structure](#project-structure)
- [The 10 Agents](#the-10-agents)
- [RAG Pipeline](#rag-pipeline)
- [Orchestration Flow](#orchestration-flow)
- [LangGraph Orchestration](#langgraph-orchestration)
- [Memory](#memory)
- [FastAPI](#fastapi)
- [Test Data](#test-data)
- [RAG vs Without RAG](#rag-vs-without-rag)
- [Known Limitations](#known-limitations)
- [What I'd Improve Next](#what-id-improve-next)
- [Installation](#installation)
- [Running the Project](#running-the-project)
- [Tests](#tests)
- [Results](#results)

---

## 🎯 Overview & Rationale

### The Project

Speny is an AI-powered multi-agent system for personal finance management.
It analyzes a user's transactions, detects anomalies, predicts future
expenses, simulates what-if scenarios, and generates personalized
recommendations in natural language.

**Example supported questions:**

- _"Have I exceeded my budget this month?"_
- _"How much will I spend next month?"_
- _"What happens if I cut my spending by 20%?"_
- _"Give me a full analysis of my finances"_

### Rationale for the Use Case

Personal finance management is particularly well-suited to a multi-agent
architecture for three reasons:

1. **Multidimensionality** — A complete financial analysis simultaneously
   requires anomaly detection, ML forecasting, behavioral analysis, goal
   tracking, and scenario simulation. No single agent can efficiently cover
   all these dimensions.

2. **Private Data & RAG** — Bank transactions are private data not present
   in LLMs. RAG is essential to ground responses in the user's real data,
   preventing any hallucination on amounts or dates.

3. **Agent Specialization** — Each aspect (budget, behavior, forecasting,
   simulation) requires distinct business logic. Splitting into specialized
   agents enables clear maintainability and natural extensibility.

---

## 🏗️ Architecture

```
User question
        ↓
┌─────────────────────────┐
│      LangGraph          │  ← Conditional routing
│   (graph.py)            │     analyzes the question
└─────────────────────────┘     decides which agents to call
        ↓
┌──────────────────────────────────────────────────┐
│                  ADVICE AGENT                    │
│            (central orchestrator)                │
└──────────────────────────────────────────────────┘
        ↓ calls as needed
┌────────┬────────┬─────────┬───────────┬──────────┬──────────┐
│ Budget │  Goal  │ Anomaly │Behavioral │  Persona │ Forecast │
│ Agent  │ Agent  │  Agent  │   Agent   │  Agent   │  Agent   │
└────────┴────────┴─────────┴───────────┴──────────┴──────────┘
        ↓ enriched by
┌──────────────────┐    ┌──────────────────┐
│  Retrieval Agent │    │ Simulation Agent │
│  (RAG + ReAct)   │    │  (scenarios)     │
└──────────────────┘    └──────────────────┘
        ↓
┌──────────────────────┐
│  Explanation Agent   │  ← Translates into plain language
└──────────────────────┘
        ↓
   Final response
```

---

## 🛠️ Tech Stack

| Component         | Technology                              |
| ----------------- | --------------------------------------- |
| Language          | Python 3.10                             |
| LLM               | Groq — llama-3.3-70b-versatile          |
| LLM Orchestration | LangChain 0.3.x + LangGraph             |
| RAG               | LlamaIndex + ChromaDB                   |
| Embeddings        | all-MiniLM-L6-v2 (SentenceTransformers) |
| ML Forecasting    | LSTM (TensorFlow/Keras)                 |
| Database          | Firebase Firestore                      |
| API               | FastAPI + Uvicorn                       |
| Agent Pattern     | ReAct (Reasoning + Acting)              |

---

## 🧭 Design Decisions

Key architectural choices made during the project and the reasoning behind them.

### Why LangGraph over a simple chain?

A basic LangChain sequential chain would call every agent on every question —
wasteful and slow. LangGraph enables **conditional routing**: a lightweight LLM
classifier first reads the question, then triggers only the relevant subgraph
(e.g., a budget question activates 2 agents, not 10). This reduced average
response time from ~90s to ~15–20s for targeted queries.

### Why RAG instead of fine-tuning?

User transaction data is private, dynamic, and user-specific — it changes every
day and is different for every user. Fine-tuning is static and would require
retraining per user. RAG grounds every LLM response in the user's actual
Firestore data at query time, making hallucination on amounts or dates
structurally impossible.

### Why ChromaDB + LlamaIndex?

ChromaDB runs locally with zero infrastructure cost, which is appropriate for a
course project and easy to reproduce. LlamaIndex handles the indexing pipeline
cleanly. Each transaction is indexed as an **atomic document** (not chunked with
surrounding context) because a transaction is a self-contained semantic unit —
amount, date, merchant, category — and fine granularity enables precise
merchant- or date-level retrieval.

### Why `all-MiniLM-L6-v2` for embeddings?

It's lightweight (384 dimensions), fast to run locally, and performs well on
short structured texts like transaction descriptions. A larger model like
`text-embedding-3-large` would add latency and cost without meaningful gain on
this data type.

### Why manual ReAct instead of `create_react_agent`?

LangChain 0.3.x does not expose `create_react_agent` in the version available
at the time of development. The ReAct loop is implemented manually via two
chained `PromptTemplate` steps (Thought → Action → Observation → Answer).
This is a known workaround and a candidate for refactoring if the dependency is
upgraded.

---

## 📁 Project Structure

```
spendwise/
├── agents/
│   ├── budget_agent.py         # Monthly budget analysis
│   ├── goal_agent.py           # Financial goal tracking
│   ├── anomaly_agent.py        # Statistical anomaly detection
│   ├── behavioral_agent.py     # Spending habit analysis
│   ├── persona_agent.py        # Financial profile (scoring)
│   ├── forecast_agent.py       # LSTM forecasting
│   ├── simulation_agent.py     # What-if scenarios
│   ├── retrieval_agent.py      # Semantic RAG + ReAct
│   ├── advice_agent.py         # Central orchestrator
│   └── explanation_agent.py    # Natural language translation
├── orchestration/
│   └── graph.py                # Conditional workflow with LangGraph
├── memory/
│   ├── short_term.py           # Session memory (RAM)
│   └── long_term.py            # Persistent memory (Firestore)
├── rag/
│   ├── indexer.py              # ChromaDB indexing
│   └── query_engine.py         # Semantic search
├── data/
│   └── firebase_client.py      # Firestore client
├── api/
│   └── main.py                 # FastAPI
├── config/
│   └── settings.py             # Configuration (.env)
├── tests/
│   └── test_agents.py          # Tests for all agents
├── .env.example                # Environment variables
└── requirements.txt            # Python dependencies
```

---

## 🤖 The 10 Agents

### 1. Budget Agent

**Role:** Analyzes current month spending and detects budget overruns.

**Tools:** Direct Firestore queries (transactions + monthly budgets)

**Logic:**

- Computes actual spending per category
- Compares against user-defined budgets
- Calculates consumption rate (%) and alerts on overruns

**Prompt:** Receives a structured summary of spending vs. budget and generates
a textual analysis highlighting at-risk categories.

---

### 2. Goal Agent

**Role:** Tracks progress toward financial goals.

**Tools:** Firestore queries (transactions, income, goals)

**Logic:**

- Calculates available monthly savings (income - expenses)
- Estimates time remaining to reach each goal
- Identifies priority goals based on progress

**Prompt:** Receives goals and available savings, generates a prioritization
plan in natural language.

---

### 3. Anomaly Agent

**Role:** Detects unusual spending through statistical analysis.

**Tools:** Firestore queries + statistical calculations (mean, standard deviation)

**Logic:**

- Calculates mean and standard deviation per merchant over historical data
- Identifies transactions exceeding 2 standard deviations
- Supports 3 modes: `monthly`, `global`, `all_months`

**Prompt:** Receives the list of detected anomalies and generates an explanation
of suspicious transactions with their context.

---

### 4. Behavioral Agent

**Role:** Analyzes financial behavior patterns.

**Tools:** Firestore queries (full transaction history)

**Logic:**

- Identifies the most expensive day of the week
- Determines the dominant category by volume and frequency
- Calculates the overall trend (% increase/decrease)
- Analyzes spending frequency per category

**Prompt:** Receives aggregated behavioral statistics and generates insights
on the user's spending habits.

---

### 5. Persona Agent

**Role:** Calculates the user's financial profile with a 0–100 score.

**Tools:** Aggregates results from Budget Agent and Goal Agent

**Logic:**

- Savings score (0–40): monthly savings rate
- Budget discipline score (0–40): adherence to defined budgets
- Regularity score (0–20): stability of spending over time
- Assigns a profile type based on the total score
- Calculates risk tolerance (low / moderate / high)

**Prompt:** Receives calculated scores and generates a personalized financial
portrait with recommendations tailored to the profile.

---

### 6. Forecast Agent

**Role:** Predicts next month's spending using an LSTM model.

**Tools:** LSTM model trained on Firestore history (TensorFlow/Keras)

**Logic:**

- Trains a separate LSTM per spending category
- Uses a 3-month sliding window for prediction
- Calculates trend (% increase/decrease) vs. the previous month
- Identifies categories at risk of budget overrun

> Note: LlamaIndex is used for loading the LSTM model.
> The message `"LLM is explicitly disabled. Using MockLLM."` at startup
> is expected and not an error.

**Prompt:** Receives per-category forecasts and generates a forward-looking
analysis with watch points for the coming month.

---

### 7. Simulation Agent

**Role:** Simulates what-if financial scenarios.

**Tools:** Financial calculations on Firestore data

**3 available scenarios:**

- `category_reduction`: impact of reducing spending in a category
- `income_increase`: impact of an income increase on savings
- `goal_achievement`: estimated time to reach a goal

**Prompt:** Receives the simulation's numerical results and generates an
explanation of the concrete impact on the financial situation.

---

### 8. Retrieval Agent _(ReAct)_

**Role:** Answers natural language questions via RAG + ReAct.

**ReAct Pattern Implemented:**

```
Question → Thought → Action → Observation → Final Answer
```

> ⚠️ Technical note: LangChain 0.3.x does not provide `create_react_agent`.
> The ReAct pattern is implemented manually via two chained PromptTemplates
> (step1 → parse → tool → step2) in `retrieval_agent.py`.

**3 available tools:**

- `RAGSearch`: semantic search in ChromaDB (506 documents)
- `FinancialSummary`: structured summary of the last 30 days from Firestore
- `BudgetStatus`: exact budget status via Budget Agent

**Prompt step1:** Analyzes the question and decides which tool to use
(Thought + Action).

**Prompt step2:** Receives the tool's observation and generates the final
response in natural language.

---

### 9. Advice Agent _(Orchestrator)_

**Role:** Aggregates all agents and generates a coherent global recommendation.

**Tools:** Directly calls the 6 specialized agents

**4 available modes:**

- `full` (~90s): Budget + Goal + Anomaly + Behavioral + Persona + Forecast
- `quick` (~20s): Budget + Goal only
- `forecast` (~30s): Forecast + Persona
- `profile` (~15s): Persona only

**Prompt:** Receives summaries from each agent and generates a coherent
synthesis with priority recommendations ranked by impact.

---

### 10. Explanation Agent

**Role:** Translates technical outputs into simple natural language.

**Tools:** Calls any agent and post-processes its response

**3 supported audiences:**

- `general public`: plain language, everyday analogies, emojis
- `expert`: financial terminology, detailed figures, trends
- `beginner`: step-by-step explanations, concept definitions

**Prompt:** Receives a raw agent output and reformulates it for the target
audience using appropriate structure and vocabulary.

---

## 🔍 RAG Pipeline

### Overview

```
Firestore (private data)
        ↓
   indexer.py
   ┌─────────────────────────────────────────┐
   │  1. Fetch data from Firestore           │
   │  2. Convert to LlamaIndex Documents     │
   │  3. Chunking per transaction            │
   │  4. Embedding generation                │
   │     (all-MiniLM-L6-v2)                  │
   │  5. Store in ChromaDB                   │
   └─────────────────────────────────────────┘
        ↓
   ChromaDB (506 documents indexed)
        ↓
   query_engine.py
   ┌─────────────────────────────────────────┐
   │  1. Embed the question                  │
   │  2. Cosine similarity search            │
   │  3. Retrieve top-k documents            │
   │  4. Inject into LLM prompt              │
   └─────────────────────────────────────────┘
        ↓
   Retrieval Agent → Contextualized response
```

### Types of Indexed Data

| Type         | Volume | Indexed Fields                                  |
| ------------ | ------ | ----------------------------------------------- |
| Transactions | ~500   | amount, date, merchant, category, description   |
| Budgets      | ~5     | category, allocated monthly amount              |
| Goals        | ~4     | name, target amount, current progress           |

### Chunking Strategy

Each transaction is indexed as an **independent atomic document**.
This choice is justified by the nature of the data:

- A transaction is a complete semantic unit (who, when, how much, where)
- Sentence-level chunking is not suited to structured tabular data
- Fine granularity enables precise search by merchant or date

**Format of an indexed document:**

```
Transaction: 450.0 MAD at Carrefour on 2026-03-15
Category: Food | Merchant: Carrefour
```

### Embedding Model

- **Model:** `all-MiniLM-L6-v2` (SentenceTransformers)
- **Dimension:** 384
- **Choice:** Lightweight, fast, multilingual, performant on short texts

### Vector Store

- **ChromaDB** in local persistent mode (`./chroma_db`)
- Cosine similarity for search
- Top-5 documents returned per query

---

## 🔄 Orchestration Flow

### How Agents Collaborate

Agent collaboration follows a two-level flow:

**Level 1 — LangGraph routes the question**

```
Question → LLM Classifier → Appropriate Node → Targeted Agents
```

**Level 2 — Advice Agent orchestrates specialized agents**

```
Advice Agent
    ├── calls Budget Agent    → budget summary
    ├── calls Goal Agent      → goals summary
    ├── calls Anomaly Agent   → anomaly summary
    ├── calls Behavioral Agent→ behavior summary
    ├── calls Persona Agent   → user profile
    ├── calls Forecast Agent  → forecasts
    └── merges 6 summaries → LLM prompt → synthetic response
```

**Level 3 — Contextual enrichment**

```
Retrieval Agent (RAG) ──→ injects transaction context
                           into any agent that requests it
```

**Level 4 — Post-processing**

```
Explanation Agent ──→ receives Advice Agent output
                       reformulates for the target audience
```

### Full Scenario Example

```
User: "Full analysis of my finances"
        ↓
LangGraph → "full" node
        ↓
Advice Agent (full mode)
    ├── Budget Agent     → "Food exceeded at 185%"
    ├── Goal Agent       → "Savings 4940 MAD/month"
    ├── Anomaly Agent    → "14 anomalies detected"
    ├── Behavioral Agent → "Tuesday most expensive day"
    ├── Persona Agent    → "Score 48/100"
    └── Forecast Agent   → "21,730 MAD forecast"
        ↓
LLM synthesizes the 6 summaries
        ↓
Explanation Agent → reformulates for the general public
        ↓
Final response in natural language
```

---

## 🕸️ LangGraph Orchestration

LangGraph implements a conditional workflow that avoids calling
all agents for every question.

```
"Budget exceeded?"     → router → budget node    (2 agents)
"Forecasts?"           → router → forecast node  (2 agents)
"Unusual spending?"    → router → anomaly node   (1 agent)
"Full analysis?"       → router → full node      (6 agents)
```

**Graph Nodes:**

| Node         | Agents Called                    |
| ------------ | -------------------------------- |
| `budget`     | Budget Agent + Goal Agent        |
| `forecast`   | Forecast Agent + Persona Agent   |
| `anomaly`    | Anomaly Agent                    |
| `behavioral` | Behavioral Agent + Persona Agent |
| `persona`    | Persona Agent                    |
| `full`       | All agents via Advice Agent      |
| `aggregator` | Results fusion + final LLM       |

---

## 🧠 Memory

### Short-Term (`memory/short_term.py`)

Stores conversation history in RAM.
Reset on each restart.

```python
session = session_manager.get_session(user_id)
session.add_user_message("How are my finances?")
session.add_assistant_message("Analyzing...")
history = session.get_history_as_text()
```

### Long-Term (`memory/long_term.py`)

Persists data in Firestore.
Survives restarts.

```python
# User preferences
save_preferences(user_id, {"mode": "quick", "audience": "general public"})

# Advice history
save_advice_to_history(user_id, question, answer, agents_used)

# Personal notes
save_user_note(user_id, "Save for a goal in December")
```

---

## 🚀 FastAPI

The API exposes all agents via REST endpoints.

**Interactive documentation:** http://127.0.0.1:8000/docs

| Method | Endpoint                    | Description                      |
| ------ | --------------------------- | -------------------------------- |
| GET    | `/health`                   | API status                       |
| POST   | `/api/advice`               | Full analysis (orchestrator)     |
| POST   | `/api/chat`                 | Free question (RAG + ReAct)      |
| GET    | `/api/budget/{user_id}`     | Budget analysis                  |
| GET    | `/api/goals/{user_id}`      | Financial goals                  |
| POST   | `/api/anomalies`            | Anomaly detection                |
| GET    | `/api/behavioral/{user_id}` | Spending habits                  |
| GET    | `/api/persona/{user_id}`    | Financial profile                |
| GET    | `/api/forecast/{user_id}`   | LSTM forecasts                   |
| POST   | `/api/simulate`             | Scenario simulation              |
| POST   | `/api/explain`              | Plain-language explanation       |
| POST   | `/api/explain/concept`      | Concept explanation              |

---

## 🗄️ Test Data

### Test User

```
USER_ID   : S6pwTrQB8R7GyuvBdyp0
Currency  : MAD (Moroccan dirhams)
Period    : March 2025 → April 2026
Volume    : ~500 transactions
```

### Data Available in Firestore

| Collection     | Contents                                               |
| -------------- | ------------------------------------------------------ |
| `transactions` | ~500 transactions (amount, date, merchant, category)   |
| `budgets`      | Food (4000 MAD/month), Education (552 MAD/month)       |
| `goals`        | Trip Japan (40%), New Laptop (60%), New House (0%)     |

### Re-indexing Data into ChromaDB

If ChromaDB is empty or needs to be rebuilt, run:

```bash
python -m rag.indexer
```

This fetches all Firestore data and rebuilds the vector index
(~506 documents, ~30 seconds).

---

## ⚖️ RAG vs Without RAG

RAG is central to Speny. Here is the concrete difference:

### Without RAG — Generic and Incorrect Response

```
Question: "Have I exceeded my budget this month?"

LLM response (without RAG):
"It is difficult to answer without knowing your financial data.
Generally, a budget is exceeded when expenses surpass the allocated
income. I recommend checking your banking application."
```

→ Useless response, no real data, no added value.

### With RAG — Precise and Contextualized Response

```
Question: "Have I exceeded my budget this month?"

Context injected by RAG (ChromaDB excerpt):
- Food budget: 4000 MAD/month
- Food spending April 2026: 7431 MAD
- Transactions: Carrefour 450 MAD, Restaurant X 320 MAD...

LLM response (with RAG):
"Yes, your Food budget is exceeded at 185.8% in April 2026.
You spent 7431 MAD against a budget of 4000 MAD, an overrun of
3431 MAD. The main expenses come from Carrefour and restaurants.
I recommend cutting back on restaurant outings in the coming weeks."
```

→ Precise, quantified, actionable response anchored in real data.

---

## ⚠️ Known Limitations

These are known constraints of the current implementation, documented for
transparency and future reference.

**LSTM forecasting on limited data.** The Forecast Agent trains a separate LSTM
per spending category on ~13 months of transactions. Some categories have fewer
than 20 data points, which is insufficient for an LSTM to generalize reliably.
The model still produces directionally useful forecasts, but they should be
treated as estimates rather than precise predictions. A simpler model like
Facebook Prophet or even a weighted moving average would likely perform better
at this data scale.

**Manual ReAct implementation.** The ReAct loop in `retrieval_agent.py` is
implemented via two chained `PromptTemplate` steps rather than a native
framework abstraction. This works correctly but is sensitive to LLM output
formatting — if the model doesn't follow the expected `Thought/Action` structure
exactly, parsing can fail. This is a known fragility that would be resolved by
upgrading to a LangChain version with native `create_react_agent` support.

**Integration tests only.** The test suite in `tests/test_agents.py` runs
agents against real Firestore and Groq API calls. This means tests require
credentials to run, are slow (~1–2 min for full suite), and cannot run in CI
without secrets. There are no unit tests with mocked dependencies.

**No error handling or retries on LLM calls.** If Groq returns a rate limit
error or a malformed response, the current implementation will raise an
unhandled exception. Production use would require retry logic and graceful
degradation.

---

## 🚀 What I'd Improve Next

Given more time, these are the highest-priority improvements:

- **Replace LSTM with Prophet for forecasting** — Better suited to short time
  series, interpretable, handles seasonality automatically, and requires no
  GPU.
- **Add unit tests with mocked dependencies** — Each agent should be testable
  in isolation using `unittest.mock` to stub Firestore and LLM calls, making
  the test suite fast and CI-friendly.
- **Upgrade LangChain and use native ReAct** — Replacing the manual two-step
  prompt chain with `create_react_agent` would make the retrieval agent more
  robust and easier to maintain.
- **Add structured error handling** — Wrap all LLM and Firestore calls in
  try/except with exponential backoff retries and fallback responses so the API
  degrades gracefully under failure.
- **Add type hints and docstrings throughout** — Currently absent in most
  agent files; adding them would significantly improve readability and
  IDE support.

---

## ⚙️ Installation

### 1. Clone the Project

```bash
git clone
cd spendwise
```

### 2. Create the Virtual Environment

```bash
python -m venv .venv
.venv\Scripts\activate      # Windows
source .venv/bin/activate   # Linux/Mac
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables

```bash
cp .env.example .env
# Fill in .env with your keys
```

The `.env` file must contain:

```bash
GROQ_API_KEY=your_groq_key_here
LLM_MODEL=llama-3.3-70b-versatile
FIREBASE_PROJECT_ID=your_firebase_project_id_here
FIREBASE_CREDENTIALS_PATH=firebase_credentials.json
CHROMA_PERSIST_DIR=.chroma_db
```

### 5. Add Firebase Credentials

Download the `firebase_credentials.json` file from the Firebase console
and place it at the project root.

### 6. Index Data into ChromaDB

```bash
python -m rag.indexer
```

---

## ▶️ Running the Project

### Start the API

```bash
uvicorn api.main:app --reload --port 8000
```

### Access the Interactive Documentation

```
http://127.0.0.1:8000/docs
```

---

## 🧪 Tests

Test a specific agent:

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

Test all agents:

```bash
python -m tests.test_agents all
```

---

## 📊 Results

Results obtained on the test user (`USER_ID = S6pwTrQB8R7GyuvBdyp0`):

| Agent            | Result                                                      |
| ---------------- | ----------------------------------------------------------- |
| Budget Agent     | Food exceeded at **185.8%** (7431 MAD / 4000 MAD)          |
| Goal Agent       | Monthly savings **4940 MAD** (rate 23.4%)                   |
| Anomaly Agent    | **14 anomalies** detected across history                    |
| Behavioral Agent | Tuesday = most expensive day, Food = **31%** of spending    |
| Persona Agent    | **"Developing"** profile, score **48/100**                  |
| Forecast Agent   | **21,730 MAD** forecast for next month (LSTM)               |
| Simulation Agent | Food -20% reduction → savings of **1,533 MAD/month**        |
| Retrieval Agent  | Semantic RAG + ReAct on **506 documents**                   |
| LangGraph        | Conditional routing — **5 possible routes**                 |
| Memory           | Short-term (RAM) + Long-term (Firestore) ✅                 |
| FastAPI          | 12 operational REST endpoints ✅                            |

---

## 🎁 Deliverables Included

This repository contains the expected deliverables:

- Complete and functional source code organized by agent.
- Detailed `README.md` with architecture, installation, and execution.
- `requirements.txt` listing all Python dependencies.
- `.env.example` API key template without sensitive values.
- Technical documentation integrated in the README.
- Demo available via the FastAPI and `tests/test_agents.py`.

---

## 👥 Authors

Project developed as part of the **Distributed AI & Multi-Agent Systems** course.
