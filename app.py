import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse # NEW: Required to serve HTML
from pydantic import BaseModel
from src.bot import HybridBot
import uvicorn

# Initialize App
app = FastAPI(title="Hybrid RAG Chatbot")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Bot (Global State)
bot_engine = HybridBot()

class QueryRequest(BaseModel):
    text: str

# --- NEW: Serve the Chat Interface at the Root URL ---
@app.get("/")
def serve_ui():
    # This grabs the index.html file from the same folder as app.py
    file_path = os.path.join(os.path.dirname(__file__), "index.html")
    if not os.path.exists(file_path):
        return {"error": "index.html not found. Please ensure it is in the hybrid_bot folder."}
    return FileResponse(file_path)

# Moved health check to /health
@app.get("/health")
def health_check():
    gpu_status = "unknown"
    if hasattr(bot_engine.llm, "model"):
        gpu_status = str(bot_engine.llm.model.device)
    return {"status": "online", "gpu": gpu_status}

@app.post("/chat")
def chat_endpoint(request: QueryRequest):
    if not request.text:
        raise HTTPException(status_code=400, detail="Query text cannot be empty")
    
    response_data = bot_engine.get_response(request.text)
    return response_data

if __name__ == "__main__":
    # Host 0.0.0.0 makes it accessible externally on the server
    uvicorn.run(app, host="0.0.0.0", port=8085)