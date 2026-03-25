import os
from dotenv import load_dotenv

# Import the FastAPI app
from app.api import app

load_dotenv()

if __name__ == "__main__":
    import uvicorn
    print("🚀 Starting IKMS Conversational Multi-Agent RAG (Feature 5)...")
    print("Server running on http://127.0.0.1:8000")
    uvicorn.run("app.api:app", 
                host="127.0.0.1", 
                port=8000, 
                reload=True)