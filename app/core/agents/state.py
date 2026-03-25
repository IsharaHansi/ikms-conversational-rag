from typing import TypedDict, Annotated, List, Dict, Optional
from langgraph.graph import add_messages
from langchain_core.messages import BaseMessage

class QAState(TypedDict):
    """Enhanced State for Conversational Multi-Turn RAG - Feature 5"""
    
    # Current input
    question: str
    answer: Optional[str]
    
    # Conversation Memory (Most Important for Feature 5)
    messages: Annotated[List[BaseMessage], add_messages]
    history: List[Dict]                     # List of previous turns for UI
    session_id: str
    conversation_summary: Optional[str]     # For long conversations later
    
    # RAG data
    context: Optional[str]
    raw_docs: Optional[List]
    
    # Agent outputs
    retrieval_traces: Optional[str]
    context_rationale: Optional[str]
    plan: Optional[str]                     # Optional future planning