from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, AIMessage
from app.core.agents.state import QAState
from app.core.agents.retrieval.vector_store import VectorStore

llm = ChatGoogleGenerativeAI(model="models/gemini-2.5-flash", temperature=0.7)
vector_store = VectorStore()

def retrieval_node(state: QAState):
    """Retrieval Agent"""
    context, docs = vector_store.retrieve(state["question"], k=6)
    
    return {
        "context": context,
        "raw_docs": docs,
        "retrieval_traces": f"Retrieved {len(docs)} chunks for query: {state['question']}"
    }

def summarization_node(state: QAState):
    """Summarization Agent - uses history + context"""
    history_text = "\n".join([
        f"Turn {h.get('turn')}: User asked: {h.get('question')} → Answer: {h.get('answer')[:300]}..." 
        for h in state.get("history", [])
    ])

    prompt = f"""You are a helpful assistant in an ongoing conversation.

Previous Conversation:
{history_text}

Current Question: {state['question']}

Relevant Context from documents:
{state.get('context', 'No context available')}

Answer the current question naturally.
If the question refers to previous messages (it, they, the method, etc.), use the conversation history to understand the reference.
Be concise and accurate. Cite key points from context when possible.
"""

    response = llm.invoke(prompt)

    return {"answer": response.content}

def verification_node(state: QAState):
    """Verification Agent - checks answer quality"""
    prompt = f"""Verify and improve this answer if needed.

Question: {state['question']}
Proposed Answer: {state.get('answer', '')}
Context: {state.get('context', '')}

Make sure the answer is:
- Accurate to the context
- Considers conversation history
- Clear and helpful

Return only the final improved answer.
"""

    response = llm.invoke(prompt)
    return {"answer": response.content}
