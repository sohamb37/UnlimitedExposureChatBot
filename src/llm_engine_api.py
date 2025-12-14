# import sys
# import os

# # Fix: Add the parent directory (root) to sys.path so we can import 'config'
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# from llm_gateway import UnifiedLLMClient
# from config import settings

# class LLMEngineAPI:
#     def __init__(self, config=settings):
#         self.config = config
#         self.llm_client = UnifiedLLMClient()

#     def generate_response(self, user_query: str, context_data: dict = None):
#         """
#         Generates a polite response.
#         If context_data (FAQ match) is found, it uses that answer.
#         If no context is found, it politely says it doesn't know.
#         """
        
#         if context_data:
#             # If we found a match in the FAQ, we ask the LLM to phrase it naturally
#             system_prompt = "You are a helpful customer support assistant for a restaurant. Use the provided Context Answer to reply to the user naturally."
#             user_prompt = f"User Query: {user_query}\n\nContext Answer: {context_data['answer']}"
#         else:
#             # Fallback for unknown questions
#             system_prompt = "You are a helpful customer support assistant."
#             user_prompt = f"User Query: {user_query}\n\nI do not have information about this in my database. Please apologize politely and ask them to call the store."

#         return self.llm_client.generate_text(system_prompt, user_prompt, temperature=0.5)

# if __name__ == "__main__":
#     engine = LLMEngineAPI()
#     print(engine.generate_response("Can you tell me a poem?", {"answer": "Salmon is $22."}))

#---------------------------------------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------------------------------------

import sys
import os

# --- PATH FIX ---
# Ensures we can import 'config' from the root directory
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.llm_gateway import UnifiedLLMClient
from src.matcher_api import MatcherAPI
from src.vector_store import VectorStore
from config import settings

matcher = MatcherAPI()
vector_db = VectorStore()

class LLMEngineAPI:
    def __init__(self):
        # Initialize all components
        self.llm_client = UnifiedLLMClient()
        self.matcher = matcher
        self.vector_db = vector_db

    def generate_response(self, user_query: str):
        print(f"\n📨 Received Query: {user_query}")
        
        # --- PATH 1: FAST & CHEAP (FAQ Match) ---
        # We check the in-memory cache first
        match_data, score = self.matcher.find_best_match(user_query)
        
        if match_data:
            print(f"⚡ FAQ Match Found! (Score: {score:.2f})")
            # OPTIMIZATION: Return the answer directly. 
            # Do NOT call the LLM here. Zero cost.
            return match_data['answer']
        
        # --- PATH 2: SLOW & SMART (RAG Fallback) ---
        print(f"📉 Low Match Score ({score:.2f}). Switching to RAG...")

        # 1. Retrieve Context from Postgres
        # We fetch the top 3 most relevant chunks
        retrieved_docs = self.vector_db.search(user_query, limit=5)
        
        if not retrieved_docs:
            return "I apologize, but I don't have enough information to answer that question properly."

        context_text = "\n\n".join(retrieved_docs) 

        MAX_CONTEXT_CHARS = 8000
        context_text = context_text[:MAX_CONTEXT_CHARS]


        # 2. Generate Answer with Context
        # We only call the API if we actually need to synthesize new information
        system_prompt = """
        You are a helpful assistant for a business. 
        Answer the user's question using ONLY the context provided below.
        If the answer is not in the context, politely say you don't know.
        """
        
        full_user_prompt = f"Context Information:\n{context_text}\n\nUser Question: {user_query}"
        
        print("🧠 Calling LLM with RAG context...")
        return self.llm_client.generate_text(system_prompt, full_user_prompt, temperature=0.3)

# Test execution
if __name__ == "__main__":
    engine = LLMEngineAPI()
    
    # Test 1: Should hit FAQ (Instant, No API call)
    print("Response:", engine.generate_response("What is RustCheck?"))
    
    # Test 2: Should hit RAG (Database Search + API call)
    print("Response:", engine.generate_response("Tell me about the formula of rust protection. How long will it last?"))