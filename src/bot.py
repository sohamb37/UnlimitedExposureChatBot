from src.matcher import SemanticMatcher
from src.llm_engine import LLMEngine
from config import SIMILARITY_THRESHOLD

class HybridBot:
    def __init__(self):
        # Initialize both engines on server startup
        self.matcher = SemanticMatcher()
        self.llm = LLMEngine()

    def get_response(self, user_query):
        # 1. Check Dictionary (Fast Path)
        print(f"\n🔍 Analyzing Query: {user_query}")
        best_answer, score = self.matcher.find_match(user_query)
        
        print(f"   Dictionary Match Score: {score:.4f}")

        # 2. Evaluate Threshold
        if score >= SIMILARITY_THRESHOLD:
            print("   ✅ Match Found! Returning Dictionary Answer.")
            return {
                "response": best_answer,
                "source": "dictionary",
                "similarity_score": score
            }
        
        # 3. Fallback to LLM (Slow Path)
        print("   ⚠️ No close match. Invoking LLM...")
        llm_response = self.llm.generate(user_query)
        
        return {
            "response": llm_response,
            "source": "llm_inference",
            "similarity_score": score
        }