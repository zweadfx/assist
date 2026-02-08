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
This project implements three core features developed during a high-intensity development sprint, specifically tailored for the modern <b>hooper</b>.

### **1. Basketball Shoe Recommendation**
* **Definition**: A feature that derives the optimal model for a <b>hooper</b> from an extensive basketball shoe database using 'RAG' (Retrieval-Augmented Generation) technology.
* **Details**: It cross-references a <b>hooper's</b> foot width, weight, and primary playstyle with technical specifications stored in **ChromaDB** to provide scientifically grounded recommendations.

### **2. Skills & Training**
* **Definition**: An intelligent coaching feature that diagnoses a <b>hooper's</b> current capabilities and generates a personalized training program to reach their target skill level.
* **Details**: Based on the reasoning capabilities of **GPT-4o**, it designs position-specific technical drills and daily workout routines for <b>hoopers</b> of all levels.

### **3. Rulebook Q&A**
* **Definition**: A knowledge extraction feature that answers complex basketball regulations through real-time search, ensuring every <b>hooper</b> plays by the book.
* **Details**: It stores actual rulebook data in a vectorized format and provides official grounds for judgment regarding specific in-game situations described by the <b>hooper</b>.

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
