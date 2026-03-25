from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from app.core.agents.state import QAState
from app.core.agents.agents import retrieval_node, summarization_node

def build_conversational_graph():
    graph = StateGraph(QAState)
    
    # Very simple flow for debugging
    graph.add_node("retrieval", retrieval_node)
    graph.add_node("answer", summarization_node)
    
    graph.add_edge(START, "retrieval")
    graph.add_edge("retrieval", "answer")
    graph.add_edge("answer", END)
    
    memory = MemorySaver()
    app = graph.compile(checkpointer=memory)
    
    return app

conversational_app = build_conversational_graph()