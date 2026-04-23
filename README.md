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
* **Agent**: `CoachAgent` (`diagnose → retrieve → generate`)
* **Definition**: A structured training generator that creates actionable **'Daily Routine Cards'** based on a **hooper's** specific weaknesses, position, and available time.
* **Details**: Retrieves specific drills from a vector database (47 drills across shooting, dribble, defense, conditioning) and orchestrates them into a progressive workout session (Warm-up → Main Drills → Cool-down). Supports equipment-aware filtering to exclude drills requiring unavailable gear.

### **2. Weekly Drill Routine (Weekly Training Planner)**
* **Agent**: `WeeklyCoachAgent` (`diagnose → plan_week → retrieve → generate`)
* **Definition**: An advanced training planner that generates **'Weekly Training Plans'** spanning 1-7 days, distributing multiple focus areas with recovery-aware scheduling.
* **Details**: Intelligently allocates focus areas across training days via LLM-based planning with round-robin fallback, retrieves drills per day from the vector database, and generates custom drill variations (`is_custom: true`) when existing drills are insufficient.

### **3. Gear Advisor (Sensory-based Recommendation)**
* **Agent**: `GearAgent` (`analyze → retrieve → generate`)
* **Definition**: A recommendation engine that matches basketball shoes based on **'Sensory Preferences'** (e.g., cushion feel, traction grip) and **'Player Archetypes'**.
* **Details**: Cross-analyzes sensory tag embeddings, player archetype matching, and signature shoe boosting across 59 shoes and 20 player profiles. Supports budget filtering with dedicated `BudgetInsufficientError` handling.

### **4. The Whistle (AI Referee & Rule Dictionary)**
* **Agent**: `JudgeAgent` (`parse → retrieve → generate`)
* **Definition**: An on-court dispute solver that provides authoritative judgments and clear definitions of complex basketball regulations (FIBA/NBA).
* **Details**: Searches vectorized rulebooks (FIBA + NBA PDFs, article-level chunking) to cite specific articles for controversial plays and serves as an instant glossary (22 terms) for technical terminology. Includes 2-level JSON retry parsing for robust LLM output handling.

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
| **Package/Quality**| **uv & Ruff** | Ensures ultra-fast dependency management and strict code standard compliance |

---

## 🏗 System Architecture
Each feature is served by a dedicated **LangGraph StateGraph agent** that follows a consistent multi-node pipeline pattern.

```
[User Request]  →  [FastAPI Endpoint]  →  [Dedicated Agent Graph]
                                                    │
                                          ┌─────────┴─────────┐
                                          │  Node 1: Parse/   │
                                          │  Analyze Input    │
                                          ├─────────┬─────────┤
                                          │  Node 2: RAG      │
                                          │  Retrieval        │
                                          ├─────────┬─────────┤
                                          │  Node 3: LLM      │
                                          │  Generation       │
                                          └─────────┬─────────┘
                                                    │
                                          [Pydantic Validated Response]
```

1. **Input Parsing & Sanitization**: Each agent validates and sanitizes user input, including prompt injection pattern blocking.
2. **Context Augmentation (RAG)**: Retrieves domain-specific knowledge from ChromaDB (drills, shoes, players, rules, glossary) with metadata filtering.
3. **Structured Generation**: LLM generates responses constrained to Pydantic schemas, with retry logic for malformed outputs.

---

## 📊 Performance Metrics

We employ an **LLM-as-Judge** evaluation pipeline (powered by `gpt-4o`) to assess the quality of our RAG responses against an expanded dataset of 50 challenging test cases (25 for Gear Advisor, 25 for The Whistle).

### **1. Gear Advisor (Shoe Recommendation)**
*Evaluated across 25 complex player archetype and budget constraint scenarios.*
- **Accuracy:** 3.56 / 5.0
- **Logical Consistency:** 4.20 / 5.0
- **Data Fidelity:** 3.04 / 5.0

### **2. The Whistle (Rule Judgment)**
*Evaluated across 25 complex regulation scenarios including intentional fouls and violations.*
- **Accuracy:** 3.40 / 5.0
- **Logical Consistency:** 3.52 / 5.0
- **Citation Appropriateness:** 2.84 / 5.0

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
# Enter required keys such as OPENAI_API_KEY in the .env file

# 4. Run the backend server
uv run uvicorn src.main:app --reload
```
