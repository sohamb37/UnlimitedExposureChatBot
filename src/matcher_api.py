import json
import numpy as np
from openai import OpenAI
# from config import DATA_FILE, OPENAI_API_KEY

# Use 'text-embedding-3-small' for a good balance of cost/speed/performance
# or 'text-embedding-3-large' for higher accuracy.
EMBEDDING_MODEL = "text-embedding-3-small"

class SemanticMatcher:
    def __init__(self):
        print("Initializing OpenAI Client...")
        self.client = OpenAI(api_key=OPENAI_API_KEY)
        
        self.kb_questions = [] # All possible user questions from JSON
        self.kb_answers = []   # Mapping questions to answers
        self.kb_embeddings = None
        
        self.load_data()

    def get_embedding(self, text):
        """Fetches embedding for a single string or a list of strings."""
        try:
            # 1. Sanitize Input (Handle both List and String)
            if isinstance(text, list):
                # If it's a list, clean every string inside it
                text = [t.replace("\n", " ") for t in text]
            else:
                # If it's a single string, clean it directly
                text = text.replace("\n", " ")

            # 2. Call OpenAI API
            response = self.client.embeddings.create(
                input=text, 
                model=EMBEDDING_MODEL
            )
            
            # 3. Return Logic
            # If input was a list, return list of embeddings
            if isinstance(text, list):
                return [data.embedding for data in response.data]
            
            # If input was a string, return single embedding
            return response.data[0].embedding
            
        except Exception as e:
            print(f"Error generating embedding: {e}")
            return []

    def load_data(self):
        """Loads JSON and pre-computes embeddings in batches."""
        try:
            with open(DATA_FILE, 'r') as f:
                data = json.load(f)
            
            # Flatten the structure
            for entry in data:
                for q in entry["questions"]:
                    self.kb_questions.append(q)
                    self.kb_answers.append(entry["answer"])
            
            total_questions = len(self.kb_questions)
            print(f"Encoding {total_questions} dictionary entries with OpenAI...")
            
            # --- BATCHING LOGIC STARTS HERE ---
            batch_size = 500  # Safe number for OpenAI tiers (can go up to 2048)
            all_embeddings = []

            # Loop from 0 to total_questions, stepping by batch_size
            for i in range(0, total_questions, batch_size):
                # Create a slice (chunk) of the list
                batch = self.kb_questions[i : i + batch_size]
                
                print(f"Processing batch {i} to {i + len(batch)}...")
                
                # Send just this chunk to the API
                batch_embeddings = self.get_embedding(batch)
                
                # Add results to our main list
                if batch_embeddings:
                    all_embeddings.extend(batch_embeddings)
            
            # Convert the full list to numpy array at the end
            self.kb_embeddings = np.array(all_embeddings)
            # ----------------------------------
            
            print("Knowledge Base encoded.")
            
        except FileNotFoundError:
            print("⚠️ FAQ file not found. Dictionary matching will be disabled.")
        except Exception as e:
            print(f"⚠️ Error loading KB: {e}")

    def find_match(self, user_query):
        """
        Returns (best_answer, score).
        If no data exists, returns (None, 0).
        """
        if self.kb_embeddings is None or len(self.kb_embeddings) == 0:
            return None, 0.0

        # 1. Get embedding for user query
        query_embedding = self.get_embedding(user_query)
        if not query_embedding:
            return None, 0.0
        
        # Convert to numpy array and reshape for matrix multiplication (1, dim)
        query_vec = np.array(query_embedding).reshape(1, -1)

        # 2. Calculate Cosine Similarity
        # OpenAI embeddings are already normalized to length 1. 
        # So, Dot Product == Cosine Similarity.
        # Calculation: (1, dim) • (n_questions, dim).T = (1, n_questions)
        scores = np.dot(query_vec, self.kb_embeddings.T)[0]

        # 3. Find the best score and index
        best_idx = np.argmax(scores)
        best_score = scores[best_idx]
        
        return self.kb_answers[best_idx], float(best_score)

if __name__ == "__main__":    
    sem = SemanticMatcher()
    query = "what is in the menu?"
    answer, score = sem.find_match(query)
        
    print(f"Match Found: {answer}")
    print(f"Confidence Score: {score}")
# print(answer)