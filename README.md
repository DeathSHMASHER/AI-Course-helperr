# Agentic AI Course Assistant

Welcome to the **Agentic AI Course Assistant**! This repository contains a Capstone Project designed to act as an intelligent, autonomous tutor for 4th-year B.Tech students studying Agentic AI concepts.

Unlike standard static large language models, this application functions as a fully **Agentic System**—incorporating retrieval logic, dynamic execution routing, self-evaluation, and robust conversation memory.

![Streamlit App Interface - 1](./streamlit_ui_1.png)
![Streamlit App Interface - 2](./streamlit_ui_2.png)

## Overview & Capabilities

The project solves the problem of providing students with 24/7 grounded academic guidance. It consists of two central layers:
1. **The Backend Engine (`agent.py`):** Uses LangGraph to map a complex execution state diagram (StateGraph).
2. **The Frontend Layer (`capstone_streamlit.py`):** Uses Streamlit to provide an intuitive, responsive interface optimized with specific resource caching mechanisms.

### Core Features
- **Retrieval-Augmented Generation (RAG):** Evaluates user queries and seamlessly retrieves exact course documentation from an embedded **ChromaDB** vector cluster.
- **Intelligent Routing:** Upon receiving a query, an initial LLM node determines the most optimal path: whether to evaluate long-term memory, dive into the knowledge base, or call an external date/time checking tool.
- **Agentic Self-Reflection:** The generated answers are immediately checked for 'faithfulness'. If the response contains hallucinations not directly found in the syllabus material (scoring below 0.7), the graph securely forces the agent to retry its response until strict factual grounding is achieved.
- **Isolated User Sessions:** Using persistent mapping (like `thread_id` UUID generation), memory state carries over perfectly in browser sessions representing true conversational flow rather than isolated API interactions.

## Technical Architecture

* **LLM Engine:** Groq's high-speed inference of the **LLaMA 3.3 (70B parameter)** model.
* **Orchestration:** Langchain and LangGraph for cyclic state progression (`StateGraph`, `MemorySaver`).
* **Vector Embeddings:** `SentenceTransformer` utilizing the `all-MiniLM-L6-v2` dimensionality model.

![VS Code Terminal Output](./vscode_terminal.png)

## Getting Started

1. Clone the repository.
2. Ensure you have the `.env` file correctly configured with your `GROQ_API_KEY`.
3. Install dependencies:
   ```bash
   pip install langchain-groq langgraph chromadb sentence-transformers streamlit python-dotenv
   ```
4. Run the Streamlit server:
   ```bash
   streamlit run capstone_streamlit.py
   ```
5. Navigate to the local URL (usually `http://localhost:8501`) provided in your terminal!
