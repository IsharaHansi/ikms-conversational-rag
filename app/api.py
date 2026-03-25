import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import os
from dotenv import load_dotenv

# Lazy import to avoid circular issues
def get_vector_store():
    from app.core.agents.retrieval.vector_store import VectorStore
    return VectorStore()

def get_conversational_app():
    from app.core.agents.graph import conversational_app
    return conversational_app

load_dotenv()

app = FastAPI(title="IKMS Conversational RAG - Feature 5")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request / Response Models
class QARequest(BaseModel):
    question: str
    session_id: Optional[str] = "default_session"

class ConversationalResponse(BaseModel):
    answer: str
    session_id: str
    history: List[dict]

@app.post("/qa/conversation")
async def conversational_qa(request: QARequest):
    try:
        conversational_app = get_conversational_app()
        vector_store = get_vector_store()

        initial_state = {
            "question": request.question,
            "history": [],
            "session_id": request.session_id,
            "messages": [],
            "answer": None,
            "context": None
        }

        result = conversational_app.invoke(
            initial_state, 
            config={"configurable": {"thread_id": request.session_id}}
        )

        return {
            "answer": result.get("answer", "Sorry, I could not generate an answer."),
            "session_id": request.session_id,
            "history": result.get("history", [])
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/index-pdf")
async def index_pdf(file_path: str):
    try:
        vector_store = get_vector_store()
        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail="PDF not found")
        num_chunks = vector_store.load_pdf(file_path)
        return {"message": f"Indexed {num_chunks} chunks successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000, reload=True)