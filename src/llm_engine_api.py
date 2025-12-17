# import sys
# import os

# # --- PATH FIX ---
# # Ensures we can import 'config' from the root directory
# sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# from src.llm_gateway import UnifiedLLMClient
# from src.matcher_api import MatcherAPI
# from src.vector_store import VectorStore
# from config import settings

# matcher = MatcherAPI()
# vector_db = VectorStore()

# class LLMEngineAPI:
#     def __init__(self):
#         # Initialize all components
#         self.llm_client = UnifiedLLMClient()
#         self.matcher = matcher
#         self.vector_db = vector_db

#     def generate_response(self, user_query: str):
#         print(f"\n📨 Received Query: {user_query}")
        
#         # --- PATH 1: FAST & CHEAP (FAQ Match) ---
#         # We check the in-memory cache first
#         match_data, score = self.matcher.find_best_match(user_query)
        
#         if match_data:
#             print(f"⚡ FAQ Match Found! (Score: {score:.2f})")
#             # OPTIMIZATION: Return the answer directly. 
#             # Do NOT call the LLM here. Zero cost.
#             return match_data['answer']
        
#         # --- PATH 2: SLOW & SMART (RAG Fallback) ---
#         print(f"📉 Low Match Score ({score:.2f}). Switching to RAG...")

#         # 1. Retrieve Context from Postgres
#         # We fetch the top 3 most relevant chunks
#         retrieved_docs = self.vector_db.search(user_query, limit=5)
        
#         if not retrieved_docs:
#             return "I apologize, but I don't have enough information to answer that question properly."

#         context_text = "\n\n".join(retrieved_docs) 

#         MAX_CONTEXT_CHARS = 8000
#         context_text = context_text[:MAX_CONTEXT_CHARS]


#         # 2. Generate Answer with Context
#         # We only call the API if we actually need to synthesize new information
#         system_prompt = """
#         You are a helpful assistant for a business. 
#         Answer the user's question using ONLY the context provided below.
#         If the answer is not in the context, politely say you don't know.
#         """
        
#         full_user_prompt = f"Context Information:\n{context_text}\n\nUser Question: {user_query}"
        
#         print("🧠 Calling LLM with RAG context...")
#         return self.llm_client.generate_text(system_prompt, full_user_prompt, temperature=0.3)

# # Test execution
# if __name__ == "__main__":
#     engine = LLMEngineAPI()
    
#     # Test 1: Should hit FAQ (Instant, No API call)
#     print("Response:", engine.generate_response("What is RustCheck?"))
    
#     # Test 2: Should hit RAG (Database Search + API call)
#     print("Response:", engine.generate_response("What are the nearest dealer around me in Whitby?"))


#########
#########
#########

import sys
import os
from typing import List, Dict

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
        
        # Configurable History Limit (Defaults to 4 if not in config)
        self.MAX_HISTORY_TURNS = getattr(settings, 'MAX_HISTORY_TURNS', 4)

    def generate_response(self, user_query: str, chat_history: List[Dict[str, str]] = None):
        print(f"\n📨 Received Query: {user_query}")
        
        # --- PATH 1: FAST & CHEAP (FAQ Match) ---
        # We check the in-memory cache first
        match_data, score = self.matcher.find_best_match(user_query)
        
        if match_data:
            print(f"⚡ FAQ Match Found! (Score: {score:.2f})")
            # OPTIMIZATION: Return the answer directly. 
            # Do NOT call the LLM here. Zero cost.
            return (match_data['answer'], "faq", score)
        
        # --- PATH 2: SLOW & SMART (RAG Fallback) ---
        print(f"📉 Low Match Score ({score:.2f}). Switching to RAG...")

        # 1. Retrieve Context from Postgres
        # We fetch the top 5 most relevant chunks (as per your code)
        retrieved_docs = self.vector_db.search(user_query, limit=5)
        
        if not retrieved_docs:
            return ("I apologize, but I don't have enough information to answer that question properly.", "rag", score)

        context_text = "\n\n".join(retrieved_docs) 

        MAX_CONTEXT_CHARS = 8000
        context_text = context_text[:MAX_CONTEXT_CHARS]

        # 2. Process Conversation History
        history_context = ""
        if chat_history:
            # Slice to get the last N turns to control costs/context
            # Note: We slice by messages here. If MAX_HISTORY_TURNS is 4, we take the last 4 messages.
            # You can adjust logic if you want 4 *pairs* (multiply by 2).
            recent_msgs = chat_history[-self.MAX_HISTORY_TURNS:]
            
            history_lines = []
            for msg in recent_msgs:
                role = msg.get('role', 'unknown').capitalize()
                content = msg.get('content', '')
                history_lines.append(f"{role}: {content}")
            
            history_context = "\n".join(history_lines)

        # 3. Generate Answer with Context
        # We only call the API if we actually need to synthesize new information
        system_prompt = """
        You are a helpful assistant for a business. 
        Answer the user's question using ONLY the provided Context Information and Conversation History.
        If the answer is not in the context, politely say you don't know.
        """
        
        full_user_prompt = f"""
Conversation History:
{history_context}

Context Information:
{context_text}

User Question: {user_query}
"""
        
        print("🧠 Calling LLM with RAG context...")
        answer = self.llm_client.generate_text(system_prompt, full_user_prompt, temperature=0.3)
        return (answer, "rag", score)

# Test execution
if __name__ == "__main__":
    engine = LLMEngineAPI()
    
    # Mock history for testing
    mock_history = [
        {"role": "user", "content": "What is the location?"},
        {"role": "assistant", "content": "We are located at 123 Main St, Whitby."}
    ]
    
    # Test 1: Should hit FAQ (Instant, No API call)
    print("Response:", engine.generate_response("What is RustCheck?"))
    
    # Test 2: Should hit RAG with history
    print("Response:", engine.generate_response("Are there any dealers near that location?", chat_history=mock_history))