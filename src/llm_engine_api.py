from llm_gateway import UnifiedLLMClient
from matcher_api import MatcherAPI
from vector_store import VectorStore
from config import settings

class LLMEngineAPI:
    def __init__(self):
        self.llm_client = UnifiedLLMClient()
        self.matcher = MatcherAPI()
        self.vector_db = VectorStore()

    def generate_response(self, user_query: str):
        print(f"\n📨 Received Query: {user_query}")
        
        # --- STEP 1: Fast & Cheap (FAQ Match) ---
        match_data, score = self.matcher.find_best_match(user_query)
        
        if match_data:
            print(f"⚡ FAQ Match Found! (Score: {score:.2f})")
            return match_data['answer']
        
        print(f"📉 Low Match Score ({score:.2f}). Switching to RAG...")

        # --- STEP 2: Slow & Smart (RAG) ---
        # 1. Retrieve context from Postgres
        retrieved_docs = self.vector_db.search(user_query, limit=3)
        context_text = "\n\n".join(retrieved_docs)

        if not context_text:
            return "I apologize, but I don't have enough information to answer that question."

        # 2. Generate Answer with Context
        system_prompt = """
        You are a helpful assistant for a business. 
        Answer the user's question using ONLY the context provided below.
        If the answer is not in the context, say you don't know.
        """
        
        full_user_prompt = f"Context:\n{context_text}\n\nQuestion: {user_query}"
        
        return self.llm_client.generate_text(system_prompt, full_user_prompt)

# Test execution
if __name__ == "__main__":
    engine = LLMEngineAPI()
    
    # Test 1: Should hit FAQ (if configured)
    print("Response:", engine.generate_response("What time do you open?"))
    
    # Test 2: Should hit RAG
    print("Response:", engine.generate_response("Tell me about the history of the founder."))