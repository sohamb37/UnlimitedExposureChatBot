import sys
import os

# Fix: Add the parent directory (root) to sys.path so we can import 'config'
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from llm_gateway import UnifiedLLMClient
from config import settings

class LLMEngineAPI:
    def __init__(self, config=settings):
        self.config = config
        self.llm_client = UnifiedLLMClient()

    def generate_response(self, user_query: str, context_data: dict = None):
        """
        Generates a polite response.
        If context_data (FAQ match) is found, it uses that answer.
        If no context is found, it politely says it doesn't know.
        """
        
        if context_data:
            # If we found a match in the FAQ, we ask the LLM to phrase it naturally
            system_prompt = "You are a helpful customer support assistant for a restaurant. Use the provided Context Answer to reply to the user naturally."
            user_prompt = f"User Query: {user_query}\n\nContext Answer: {context_data['answer']}"
        else:
            # Fallback for unknown questions
            system_prompt = "You are a helpful customer support assistant."
            user_prompt = f"User Query: {user_query}\n\nI do not have information about this in my database. Please apologize politely and ask them to call the store."

        return self.llm_client.generate_text(system_prompt, user_prompt, temperature=0.5)

if __name__ == "__main__":
    engine = LLMEngineAPI()
    print(engine.generate_response("Can you tell me a poem?", {"answer": "Salmon is $22."}))