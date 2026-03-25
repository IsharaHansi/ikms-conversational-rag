# IKMS Conversational Multi-Agent RAG

**Feature 5: Conversational Multi-Turn QA with Memory**

An intelligent conversational Retrieval-Augmented Generation (RAG) system that maintains full conversation history and understands follow-up questions using LangGraph.

---

## ✨ Key Features

- **Multi-turn Conversation Memory** (Main requirement of Feature 5)
- Understands references like "it", "the method", "they", "its advantages"
- Multi-Agent pipeline: Retrieval Agent → Summarization Agent
- Persistent session management (New Chat support)
- Pinecone vector database for document retrieval
- Modern dark-themed Chat UI with loading animation
- FastAPI backend with CORS support

---

🛠 Technology Stack

LLM: Google Gemini 2.5 Flash
Framework: LangChain + LangGraph
Vector Store: Pinecone
Embeddings: Gemini text-embedding-004
Backend: FastAPI
Frontend: HTML + Tailwind CSS + JavaScript
Memory: LangGraph MemorySaver (Checkpointer)

---


```bash
git clone https://github.com/YOUR-USERNAME/ikms-conversational-rag.git
cd ikms-conversational-rag
