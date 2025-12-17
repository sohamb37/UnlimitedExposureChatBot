from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from src.llm_engine_api import LLMEngineAPI

app = FastAPI(title="Hybrid Restaurant Bot")

# 🔴 THIS IS CRITICAL
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # OK for demo
    allow_credentials=True,
    allow_methods=["*"],          # MUST include OPTIONS
    allow_headers=["*"],
)

engine = LLMEngineAPI()

class ChatRequest(BaseModel):
    text: str

class ChatResponse(BaseModel):
    response: str
    source: str
    similarity_score: float

@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    answer, source, score = engine.generate_response(
        user_query=req.text
    )

    return {
        "response": answer,
        "source": source,
        "similarity_score": score
    }

@app.get("/")
def health():
    return {"status": "ok"}
