<div align="center">
  <img src="https://github.com/user-attachments/assets/24896d45-950b-4deb-a190-b5782f9b12c3" width="500" alt="assist_logo">
  <br/>
  <br/>

  <img src="https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square&logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/FastAPI-005571?style=flat-square&logo=fastapi&logoColor=white"/>
  <img src="https://img.shields.io/badge/LangGraph-Agentic-orange?style=flat-square"/>
  <img src="https://img.shields.io/badge/uv-Package%20Manager-blue?style=flat-square"/>
  <img src="https://img.shields.io/badge/License-MIT-green?style=flat-square"/>

  <br/>
  
  # Assist: AI-powered Basketball Partner
  
  <p>
    <b>Assist</b> is an intelligent agent project that combines 'LangGraph' and 'RAG' technologies to help <b>hoopers</b> enhance their performance through data-driven insights.
  </p>
</div>

**🌐 Live Demo: [https://assist-frontend-plum.vercel.app](https://assist-frontend-plum.vercel.app)**

---

## 🏀 Introduction
**'Assist'** is designed to bridge the information gap faced by **hoopers** by increasing the accessibility of professional coaching systems. The AI precisely analyzes a **hooper's** physical attributes and the context of their inquiries to provide an experience akin to having a personal coach standing right by the courtside.

> "Basketball is a game of details. Assist helps **hoopers** master those details."

---

## 🚀 Key Features (MVP)
This project implements four core features, each powered by a dedicated LangGraph agent with its own state machine workflow.

### **1. AI Skill Lab (Personalized Skill Trainer)**
* **Agent**: `CoachAgent` (`diagnose → generate`)
* **Definition**: A micro-step skill breakdown generator that creates actionable **'Skill Breakdown Cards'** tailored to a **hooper's** category, skill level, and available time.
* **Details**: Uses LLM-based generation to produce 3–5 progressive steps for a single basketball skill, each adding exactly one layer of complexity (stationary → ball → movement → game-speed → game situation). Parses optional free-text input to enrich user preferences (intensity, additional focus, special notes). Enforces a strict time budget so all step durations sum to exactly the requested practice time.

### **2. Weekly Drill Routine (Weekly Training Planner)**
* **Agent**: `WeeklyCoachAgent` (`diagnose → plan_week → generate`)
* **Definition**: An advanced training planner that generates **'Weekly Training Plans'** spanning 1–7 days, distributing multiple focus areas with recovery-aware scheduling.
* **Details**: The `plan_week` node uses an LLM to intelligently allocate focus areas across training days (round-robin fallback on failure), then the `generate` node produces a complete per-day routine with structured warm-up / main / cool-down phases. All drills are LLM-generated with concrete rep/set targets and coaching tips. Supports Korean and English output via a language flag.

### **3. Gear Advisor (Sensory-based Recommendation)**
* **Agent**: `GearAgent` (`analyze → retrieve → generate`)
* **Definition**: A recommendation engine that matches basketball shoes based on **'Sensory Preferences'** (e.g., cushion feel, traction grip) and **'Player Archetypes'**.
* **Details**: Cross-analyzes sensory tag embeddings, player archetype matching, and signature shoe boosting across 59 shoes and 20 player profiles. Supports budget filtering with dedicated `BudgetInsufficientError` handling.

### **4. The Whistle (AI Referee & Rule Dictionary)**
* **Agent**: `JudgeAgent` (`parse → extract_keywords → retrieve → generate`)
* **Definition**: An on-court dispute solver that provides authoritative judgments and clear definitions of complex basketball regulations (FIBA/NBA).
* **Details**: An `extract_keywords` node distills the natural-language situation into 3–5 violation-type keywords before querying ChromaDB, narrowing the embedding space mismatch between long situation text and short rule articles. Hybrid search combines rule retrieval (FIBA + NBA PDFs, article-level chunking) with glossary lookup (22 terms). Includes 2-level JSON retry parsing for robust LLM output handling.

---

## 🔐 Authentication & Plan Storage

### **JWT Authentication**
User accounts are managed with JWT access + refresh tokens via `/api/v1/auth`. Passwords are hashed with bcrypt. Tokens are signed using a configurable `SECRET_KEY` and `HS256` algorithm.

### **Saved Plans**
Authenticated users can persist and track their training plans via `/api/v1/plans`. Each `SavedPlan` record stores the plan type (`weekly` | `skill`), training dates, and a `completed_days` array for progress tracking. Plans are stored in a SQLite (local) or PostgreSQL (production) database managed by SQLAlchemy ORM.

---

## 🛠 Tech Stack
The following technical ecosystem was established to ensure system stability and scalability for the **Assist** platform.

| Category | Technology | Rationale |
| :--- | :--- | :--- |
| **Language** | **Python 3.10+** | Provides optimized compatibility with AI and data analysis libraries |
| **Backend** | **FastAPI** | Implements high-performance API services through asynchronous processing |
| **Frontend** | **Next.js 15 + Tailwind CSS** | Delivers a responsive UI with server-side rendering and utility-first styling |
| **Orchestration** | **LangGraph** | Enables advanced agent control via state-based cyclic logic for multi-functional tasks |
| **Vector DB** | **ChromaDB** | Supports rapid data embedding and efficient vector similarity search |
| **Relational DB** | **SQLite / PostgreSQL + SQLAlchemy** | Persists user accounts and saved training plans with ORM-managed migrations |
| **Auth** | **JWT (python-jose) + bcrypt** | Stateless access/refresh token authentication with secure password hashing |
| **Package/Quality**| **uv & Ruff** | Ensures ultra-fast dependency management and strict code standard compliance |

---

## 🏗 System Architecture
Each feature is served by a dedicated **LangGraph StateGraph agent** that follows a consistent multi-node pipeline pattern.

```
[User Request]  →  [FastAPI Endpoint]  →  [Dedicated Agent Graph]
                                                    │
                           ┌────────────────────────┴────────────────────────┐
                           │                                                  │
                  [Skill / Weekly]                                  [Gear / Whistle]
                           │                                                  │
               ┌───────────┴───────────┐                      ┌──────────────┴─────────────┐
               │  Node 1: Diagnose /   │                      │  Node 1: Analyze / Parse   │
               │  Parse Input          │                      │  Input                     │
               ├───────────┬───────────┤                      ├──────────────┬─────────────┤
               │  Node 2:  │ (Weekly)  │                      │  Node 2: RAG │ (Whistle)   │
               │  LLM      │ Plan Week │                      │  Retrieval   │ +Keywords   │
               │  Generate │ → Generate│                      ├──────────────┬─────────────┤
               └───────────┴───────────┘                      │  Node 3: LLM │             │
                           │                                  │  Generation  │             │
                           └──────────────────────────────────┘─────────────┘
                                                    │
                                          [Pydantic Validated Response]
```

1. **Input Parsing & Sanitization**: Each agent validates and sanitizes user input, including prompt injection pattern blocking and field-length enforcement.
2. **Context Augmentation (RAG)**: Gear and Whistle agents retrieve domain-specific knowledge from ChromaDB (shoes, players, rules, glossary) with metadata filtering. Whistle additionally runs an LLM keyword extraction step before vector search.
3. **LLM Generation**: All agents use `gpt-4o` for final generation against Pydantic schemas. Skill and Weekly agents use pure LLM generation without RAG; Whistle uses `gpt-4o-mini` for keyword extraction to reduce cost.
4. **Structured Output with Retry**: Responses are validated against Pydantic schemas. Whistle implements a 2-level retry loop (re-send invalid JSON to LLM for correction before raising an error).

---

## 📊 Performance Metrics

We employ a two-layer evaluation pipeline to assess RAG quality separately for retrieval and generation.

### **1. Gear Advisor (Shoe Recommendation)**
*Evaluated across 25 complex player archetype and budget constraint scenarios.*

| Layer | Metric | Value |
| :--- | :--- | :--- |
| **Retrieval** | Hit@3 | **0.92** |
| **Retrieval** | MRR | **0.87** |
| **Generation** | Accuracy (LLM Judge) | **4.36 / 5.0** |
| **Generation** | Data Fidelity (LLM Judge) | **4.76 / 5.0** |

### **2. The Whistle (Rule Judgment)**
*Evaluated across 25 complex regulation scenarios including intentional fouls and violations.*

| Layer | Metric | Value |
| :--- | :--- | :--- |
| **Retrieval** | Rule Hit@3 | **0.56** |
| **Retrieval** | Rule MRR | **0.46** |
| **Generation** | Citation Hit Rate | **0.72** |
| **Generation** | Accuracy (LLM Judge) | **4.68 / 5.0** |
| **Generation** | Citation (LLM Judge) | **4.40 / 5.0** |
| **Generation** | Faithfulness (LLM Judge) | **4.76 / 5.0** |

### **3. AI Skill Lab (Skill Breakdown)**
*Evaluated for constraint compliance across multiple skill level and equipment scenarios.*

| Metric | Description |
| :--- | :--- |
| **Equipment Pass Rate** | Fraction of cases where generated steps respect available equipment |
| **Time Pass Rate** | Fraction of cases where step durations sum exactly to the requested time |

To run evaluations:
```bash
uv run python scripts/evaluate.py --all      # all agents
uv run python scripts/evaluate.py --gear     # Gear Advisor only
uv run python scripts/evaluate.py --whistle  # The Whistle only
uv run python scripts/evaluate.py --skill    # Skill Lab only
```

---

## 💻 Getting Started

### **Prerequisites**
* Python 3.10 or higher
* **uv** package manager installed

### **Installation & Run**
```bash
# 1. Clone the repository
git clone https://github.com/zweadfx/assist.git
cd assist

# 2. Install dependencies and sync virtual environment
uv sync

# 3. Configure environment variables
cp .env.example .env
# Enter required keys in the .env file:
#   OPENAI_API_KEY=...
#   SECRET_KEY=...          (any random string for JWT signing)
#   DATABASE_URL=...        (defaults to sqlite:///./data/assist.db)

# 4. Run the backend server
uv run uvicorn src.main:app --reload
```
