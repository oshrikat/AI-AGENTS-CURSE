# Interactive AI Financial Assistant (LangGraph & RAG)

## 📌 Overview
This repository contains an evolving academic project focused on building an interactive, educational AI agent. The core goal of this assistant is to help users understand financial portfolios and specific stocks through a guided, objective, and conversational interface. 

The project is developed in iterative stages, gradually upgrading a basic script into a sophisticated, multi-intent state machine.

## 🚀 Features by Stage

### ✅ Stage 1 & 2: Linear Logic & Basic LangGraph
* **Interactive CLI:** Built a simple interface to collect user preferences (domain, risk, horizon).
* **Linear State Machine:** Implemented a basic LangGraph pipeline with a straight-line execution (`AgentState` -> `Collection` -> `Filtering` -> `LLM Summary`).

### ✅ Stage 3: Retrieval-Augmented Generation (RAG)
* **Semantic Search:** Replaced hardcoded `if/else` filtering with a vector-based search.
* **Vector Database:** Utilized **ChromaDB** to store 25 stock profiles.
* **Embeddings:** Integrated Google's Gemini Embedding model (`gemini-embedding-2`) to match user queries with the most semantically relevant stock profiles.

### ✅ Stage 4: Multi-Intent Routing & Clarification Loops
* **Dynamic Routing:** Upgraded the agent to a multi-branch state machine using LangGraph's conditional edges.
* **Intent Classification:** The agent analyzes free-text input and routes to either:
  * `explain_stock`: Direct RAG retrieval for a specific stock.
  * `suggest_portfolio`: A guided portfolio building process.
  * `unknown`: A fallback loop requesting the user to rephrase.
* **Clarification Loops:** If a user requests a portfolio but misses required details (like budget or risk), the agent dynamically loops back to ask for the missing data before proceeding to the vector search.

### ✅ Stage 5: Structured Output & Robust Routing
* **Pydantic Schemas:** Replaced brittle string parsing with strictly typed data models (`IntentResult` and `RequirementsResult`).
* **Structured LLM Output:** Integrated LangChain's `with_structured_output()` to force the LLM to return reliable, machine-readable objects instead of free-form text.
* **Deterministic Conditional Routing:** Graph transitions, clarification loops, and branches are now driven dynamically by extracted object properties (e.g., arrays of missing fields), ensuring 100% stable control flow.


### 🚧 Upcoming Stages
*(This section will be updated as the course progresses and new features are added)*

## 🛠️ Tech Stack
* **Language:** Python
* **Orchestration:** LangGraph / LangChain
* **Vector Store:** ChromaDB (Local SQLite/Binary storage)
* **LLM & Embeddings:** Google GenAI API (Gemini 2.5 Flash, Gemini Embeddings)

## ⚠️ Disclaimer
This is an educational project created for academic purposes. The AI assistant does not provide real financial advice.