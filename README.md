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

---

## 🏀 Introduction
**'Assist'** is designed to bridge the information gap faced by <b>hoopers</b> by increasing the accessibility of professional coaching systems. The AI precisely analyzes a <b>hooper's</b> physical attributes and the context of their inquiries to provide an experience akin to having a personal coach standing right by the courtside.

> "Basketball is a game of details. Assist helps <b>hoopers</b> master those details."

---

## 🚀 Key Features (MVP)
This project implements three core features developed during a high-intensity development sprint, specifically tailored for the modern **hooper**.

### **1. AI Skill Lab (Personalized Skill Trainer)**
* **Definition**: A structured training generator that creates actionable **'Daily Routine Cards'** based on a **hooper's** specific weaknesses, position, and available time.
* **Details**: Instead of generic advice, it retrieves specific drills from a vector database and orchestrates them into a complete workout session (Warm-up → Main Drills → Cool-down) in a checklist format to ensure immediate court application.

### **2. Gear Advisor (Sensory-based Recommendation)**
* **Definition**: A next-gen recommendation engine that matches basketball shoes based on **'Sensory Preferences'** (e.g., cushion feel, traction sound) and **'Player Archetypes'**.
* **Details**: It goes beyond basic specs by analyzing subjective inputs (like "I want to move like Kyrie") alongside physical constraints (wide feet, injuries) to find the perfect fit using RAG technology.

### **3. The Whistle (AI Referee & Rule Dictionary)**
* **Definition**: An on-court dispute solver that provides authoritative judgments and clear definitions of complex basketball regulations (FIBA/NBA/KBL).
* **Details**: It acts as a real-time judge by searching vectorized rulebooks to cite specific articles for controversial plays (e.g., traveling vs. gather step) and serves as an instant glossary for technical terms.
---

## 🛠 Tech Stack
The following technical ecosystem was established to ensure system stability and scalability for the **Assist** platform.

| Category | Technology | Rationale |
| :--- | :--- | :--- |
| **Language** | **Python 3.10+** | Provides optimized compatibility with AI and data analysis libraries |
| **Backend** | **FastAPI** | Implements high-performance API services through asynchronous processing |
| **Orchestration** | **LangGraph** | Enables advanced agent control via state-based cyclic logic for multi-functional tasks |
| **Vector DB** | **ChromaDB** | Supports rapid data embedding and efficient vector similarity search |
| **Package/Quality**| **uv & Ruff** | Ensures ultra-fast dependency management and strict code standard compliance |

---

## 🏗 System Architecture
**Assist** adopts an agent architecture characterized by organic state transitions between a 'Router' and various 'Task Nodes' to serve the <b>hooper's</b> needs.



1.  **Intent Routing**: Analyzes the input to allocate the task to the most suitable node among 'Recommendation', 'Training', or 'Rules'.
2.  **Context Augmentation**: Extracts necessary external knowledge (RAG) during the 'Retrieval' phase to increase prompt accuracy.
3.  **Self-Correction**: Performs an iterative reasoning process to self-verify and correct any logical errors in the generated response before reaching the <b>hooper</b>.

---

## 💻 Getting Started

### **Prerequisites**
* Python 3.10 or higher
* **uv** package manager installed

### **Installation & Run**
```bash
# 1. Clone the repository
git clone [https://github.com/your-username/assist.git](https://github.com/your-username/assist.git)
cd assist

# 2. Install dependencies and sync virtual environment
uv sync

# 3. Configure environment variables
cp .env.example .env
# Enter required keys such as OPENAI_API_KEY in the .env file

# 4. Run the development server
uv run uvicorn src.main:app --reload
