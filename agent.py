import os
import datetime
from typing import TypedDict, List
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
import chromadb
from sentence_transformers import SentenceTransformer

# 1. Knowledge Base Documents
DOCUMENTS = [
    {
        "id": "doc_001",
        "topic": "Agentic AI Overview",
        "text": "Agentic AI refers to systems that can autonomously pursue goals, make decisions, and use tools to interact with their environment. Unlike passive QA systems, an agent has routing capabilities and can determine whether it needs to search a knowledge base, perform a calculation, or simply respond from memory to achieve the user's objective."
    },
    {
        "id": "doc_002",
        "topic": "LangGraph Basics",
        "text": "LangGraph is a library for building stateful, multi-actor applications with LLMs. It uses graph-based architectures where each node represents a processing step (like retrieving context or generating an answer) and edges represent the flow of data. Cyclic graphs allow for reflection loops, where the agent can evaluate its own response and retry if necessary."
    },
    {
        "id": "doc_003",
        "topic": "StateGraph and TypedDict",
        "text": "In LangGraph, the StateGraph requires a unified State object, commonly defined as a Python TypedDict. Every node in the graph accepts this state as input and returns updates. If a node updates a field (like appending a new message), that field must exist in the State definition, otherwise runtime KeyErrors will occur."
    },
    {
        "id": "doc_004",
        "topic": "ChromaDB for Implementation",
        "text": "ChromaDB is an open-source vector database used for Retrieval-Augmented Generation (RAG). It stores text documents alongside their numerical embeddings. When a user asks a question, ChromaDB compares the question's embedding to the stored document embeddings to quickly retrieve the most semantically relevant chunks of information."
    },
    {
        "id": "doc_005",
        "topic": "Memory and Sliding Windows",
        "text": "LLMs lack inherent memory. Context must be passed explicitly in every API call. LangGraph's MemorySaver persists graph states across sessions using a thread_id. To prevent token limit overflow on free tiers (like Groq), a sliding window approach is often used, maintaining only the last few message turns rather than the entire infinite history."
    },
    {
        "id": "doc_006",
        "topic": "Tool Use Beyond Retrieval",
        "text": "Agents can use external tools to answer questions that a static knowledge base cannot cover. Tools can range from web search, calculators, and API routers, to real-time clock functions. When building a tool node, any exceptions must be handled elegantly—returning error strings instead of crashing, ensuring the graph run completes."
    },
    {
        "id": "doc_007",
        "topic": "Routing Logic",
        "text": "A router node directs the flow of execution based on the user's query. It asks the LLM to classify the query—for instance, choosing between 'retrieve' for domain specifics, 'memory_only' for conversational follow-ups, or 'tool' for dynamic data. This minimizes unnecessary database queries and reduces latency."
    },
    {
        "id": "doc_008",
        "topic": "Self-Reflection and Faithfulness",
        "text": "To prevent hallucinations, agents can evaluate their own answers. An evaluation node scores the 'faithfulness' of a generated response by checking if it strictly adheres to the retrieved context. If the score falls below a set threshold (e.g., 0.7), the graph can loop back, instructing the answer node to retry and adhere strictly to facts."
    },
    {
        "id": "doc_009",
        "topic": "RAGAS Evaluation Framework",
        "text": "RAGAS (Retrieval Augmented Generation Assessment) is a framework for evaluating RAG pipelines without human annotation. It generates baseline quality metrics such as Faithfulness (is the answer grounded?), Answer Relevancy (does it answer the question?), and Context Precision (was the retrieved data useful?)."
    },
    {
        "id": "doc_010",
        "topic": "Deployment with Streamlit",
        "text": "Streamlit is a Python framework for rapidly creating interactive web UI dashboards. For Agentic AI, heavy initializations like embedding models and ChromaDB clients should be cached using @st.cache_resource. This prevents the models from reloading on every interaction, preserving chat memory and saving processing power."
    }
]

# 2. State Design
class CapstoneState(TypedDict):
    question: str
    messages: List[dict]
    route: str
    retrieved: str
    sources: List[str]
    tool_result: str
    answer: str
    faithfulness: float
    eval_retries: int

FAITHFULNESS_THRESHOLD = 0.7
MAX_EVAL_RETRIES = 2

# 3. Agent Builder Class
class AgentBuilder:
    def __init__(self):
        self.llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)
        self.embedder = SentenceTransformer("all-MiniLM-L6-v2")
        self.collection = self._init_chroma()
        self.app = self._build_graph()

    def _init_chroma(self):
        client = chromadb.Client()
        try:
            client.delete_collection("capstone_kb")
        except:
            pass
        collection = client.create_collection("capstone_kb")
        texts = [d["text"] for d in DOCUMENTS]
        ids = [d["id"] for d in DOCUMENTS]
        embeddings = self.embedder.encode(texts).tolist()
        collection.add(
            documents=texts,
            embeddings=embeddings,
            ids=ids,
            metadatas=[{"topic": d["topic"]} for d in DOCUMENTS]
        )
        return collection

    def memory_node(self, state: CapstoneState) -> dict:
        msgs = state.get("messages", [])
        msgs = msgs + [{"role": "user", "content": state["question"]}]
        if len(msgs) > 6:
            msgs = msgs[-6:]
        return {"messages": msgs}

    def router_node(self, state: CapstoneState) -> dict:
        question = state["question"]
        messages = state.get("messages", [])
        recent = "; ".join(f"{m['role']}: {m['content'][:60]}" for m in messages[-3:-1]) or "none"

        prompt = f"""You are a router for a chatbot acting as a Course Assistant for an Agentic AI course.

Available options:
- retrieve: search the knowledge base for syllabus or Agentic AI concepts
- memory_only: answer from conversation history (e.g. 'what did you just say?')
- tool: use the datetime tool ONLY when asked about the current date, today, or time

Recent conversation: {recent}
Current question: {question}

Reply with ONLY one word: retrieve / memory_only / tool"""
        
        response = self.llm.invoke(prompt)
        decision = response.content.strip().lower()
        if "memory" in decision: decision = "memory_only"
        elif "tool" in decision: decision = "tool"
        else: decision = "retrieve"
        return {"route": decision}

    def retrieval_node(self, state: CapstoneState) -> dict:
        q_emb = self.embedder.encode([state["question"]]).tolist()
        results = self.collection.query(query_embeddings=q_emb, n_results=3)
        chunks = results["documents"][0]
        topics = [m["topic"] for m in results["metadatas"][0]]
        context = "\n\n---\n\n".join(f"[{topics[i]}]\n{chunks[i]}" for i in range(len(chunks)))
        return {"retrieved": context, "sources": topics}

    def skip_retrieval_node(self, state: CapstoneState) -> dict:
        return {"retrieved": "", "sources": []}

    def tool_node(self, state: CapstoneState) -> dict:
        try:
            today = datetime.date.today().strftime("%B %d, %Y")
            tool_result = f"Current server date: {today}"
        except Exception as e:
            tool_result = f"Date check error: {str(e)}"
        return {"tool_result": tool_result}

    def answer_node(self, state: CapstoneState) -> dict:
        question = state["question"]
        retrieved = state.get("retrieved", "")
        tool_result = state.get("tool_result", "")
        messages = state.get("messages", [])
        eval_retries = state.get("eval_retries", 0)

        context_parts = []
        if retrieved: context_parts.append(f"KNOWLEDGE BASE:\n{retrieved}")
        if tool_result: context_parts.append(f"TOOL RESULT:\n{tool_result}")
        context = "\n\n".join(context_parts)

        if context:
            system_content = f"""You are a helpful Course Assistant for 4th year B.Tech students studying Agentic AI.
Answer using ONLY the information provided in the context below.
If the answer is not in the context, say: I don't have that information in my knowledge base.
Do NOT add information from your training data or hallucinate technical details outside the class material.

{context}"""
        else:
            system_content = """You are a helpful Course Assistant. Answer based on the conversation history."""

        if eval_retries > 0:
            system_content += "\n\nIMPORTANT: Your previous answer did not meet quality standards. Answer using ONLY information explicitly stated in the context above."

        lc_msgs = [SystemMessage(content=system_content)]
        for msg in messages[:-1]:
            lc_msgs.append(HumanMessage(content=msg["content"]) if msg["role"] == "user" else AIMessage(content=msg["content"]))
        lc_msgs.append(HumanMessage(content=question))

        response = self.llm.invoke(lc_msgs)
        return {"answer": response.content}

    def eval_node(self, state: CapstoneState) -> dict:
        answer = state.get("answer", "")
        context = state.get("retrieved", "")[:500]
        retries = state.get("eval_retries", 0)

        if not context:
            return {"faithfulness": 1.0, "eval_retries": retries + 1}

        prompt = f"""Rate faithfulness: does this answer use ONLY information from the context?
Reply with ONLY a number between 0.0 and 1.0.
1.0 = fully faithful. 0.5 = some hallucination. 0.0 = mostly hallucinated.

Context: {context}
Answer: {answer[:300]}"""
        
        result = self.llm.invoke(prompt).content.strip()
        try:
            score = float(result.split()[0].replace(",", "."))
            score = max(0.0, min(1.0, score))
        except:
            score = 0.5
        return {"faithfulness": score, "eval_retries": retries + 1}

    def save_node(self, state: CapstoneState) -> dict:
        messages = state.get("messages", [])
        messages = messages + [{"role": "assistant", "content": state["answer"]}]
        return {"messages": messages}

    def route_decision(self, state: CapstoneState) -> str:
        route = state.get("route", "retrieve")
        if route == "tool": return "tool"
        if route == "memory_only": return "skip"
        return "retrieve"

    def eval_decision(self, state: CapstoneState) -> str:
        score = state.get("faithfulness", 1.0)
        retries = state.get("eval_retries", 0)
        if score >= FAITHFULNESS_THRESHOLD or retries >= MAX_EVAL_RETRIES:
            return "save"
        return "answer"

    def _build_graph(self):
        graph = StateGraph(CapstoneState)
        graph.add_node("memory", self.memory_node)
        graph.add_node("router", self.router_node)
        graph.add_node("retrieve", self.retrieval_node)
        graph.add_node("skip", self.skip_retrieval_node)
        graph.add_node("tool", self.tool_node)
        graph.add_node("answer", self.answer_node)
        graph.add_node("eval", self.eval_node)
        graph.add_node("save", self.save_node)

        graph.set_entry_point("memory")
        graph.add_edge("memory", "router")
        graph.add_conditional_edges("router", self.route_decision, {"retrieve": "retrieve", "skip": "skip", "tool": "tool"})
        graph.add_edge("retrieve", "answer")
        graph.add_edge("skip", "answer")
        graph.add_edge("tool", "answer")
        graph.add_edge("answer", "eval")
        graph.add_conditional_edges("eval", self.eval_decision, {"answer": "answer", "save": "save"})
        graph.add_edge("save", END)

        checkpointer = MemorySaver()
        return graph.compile(checkpointer=checkpointer)

def load_agent():
    builder = AgentBuilder()
    return builder.app, builder.embedder, builder.collection
