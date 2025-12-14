import json
import os
import pickle
import numpy as np
from llm_gateway import UnifiedLLMClient
from config import settings

class MatcherAPI:
    def __init__(self):
        self.client = UnifiedLLMClient()
        self.faq_data = self._load_faq()
        self.faq_embeddings = []
        # Define the cache path in the data folder
        self.embeddings_cache_path = os.path.join("data", "faq_embeddings.pkl")
        self._precompute_embeddings()

    def _load_faq(self):
        if not os.path.exists(settings.FAQ_FILE_PATH):
            return []
        with open(settings.FAQ_FILE_PATH, 'r', encoding='utf-8') as f:
            return json.load(f)

    def _precompute_embeddings(self):
        """
        Loads embeddings for all FAQ questions.
        Checks for a cached pickle file first to avoid re-calculating on every restart.
        """
        cache_valid = False
        
        # Check if cache exists and is newer than the source FAQ file
        if os.path.exists(self.embeddings_cache_path) and os.path.exists(settings.FAQ_FILE_PATH):
            faq_mtime = os.path.getmtime(settings.FAQ_FILE_PATH)
            cache_mtime = os.path.getmtime(self.embeddings_cache_path)
            if cache_mtime > faq_mtime:
                cache_valid = True

        if cache_valid:
            print(f"⚙️  Loading FAQ embeddings from cache: {self.embeddings_cache_path}")
            try:
                with open(self.embeddings_cache_path, 'rb') as f:
                    self.faq_embeddings = pickle.load(f)
                
                # Sanity check: Ensure cache matches current data length
                if len(self.faq_embeddings) == len(self.faq_data):
                    print("✅ Matcher Ready (Loaded from Cache).")
                    return
                else:
                    print("⚠️  Cache mismatch (data length differs). Recomputing...")
            except Exception as e:
                print(f"⚠️  Error loading cache: {e}. Recomputing...")

        # If cache is missing, stale, or invalid, compute from scratch
        print("⚙️  Computing FAQ embeddings (Cache miss or stale)...")
        self.faq_embeddings = [] 
        
        for item in self.faq_data:
            # We embed the first question variation as the 'anchor'
            # For better accuracy, you could embed all variations.
            anchor_question = item['questions'][0]
            vector = self.client.get_embedding(anchor_question)
            self.faq_embeddings.append(vector)
            
        # Save the new embeddings to cache
        try:
            # Ensure the directory exists
            os.makedirs(os.path.dirname(self.embeddings_cache_path), exist_ok=True)
            with open(self.embeddings_cache_path, 'wb') as f:
                pickle.dump(self.faq_embeddings, f)
            print(f"💾 Saved embeddings to cache: {self.embeddings_cache_path}")
        except Exception as e:
            print(f"⚠️  Could not save cache: {e}")

        print("✅ Matcher Ready.")

    def _cosine_similarity(self, v1, v2):
        return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

    def find_best_match(self, user_query: str):
        """
        Returns (answer, score).
        """
        if not self.faq_data:
            return None, 0.0

        query_vector = self.client.get_embedding(user_query)
        best_score = -1
        best_idx = -1

        for i, faq_vector in enumerate(self.faq_embeddings):
            score = self._cosine_similarity(query_vector, faq_vector)
            if score > best_score:
                best_score = score
                best_idx = i

        # Check Threshold
        if best_score >= settings.FAQ_SIMILARITY_THRESHOLD:
            return self.faq_data[best_idx], best_score
        
        return None, best_score