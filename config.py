import os
import torch

# --- Settings ---

# 1. Similarity Threshold (0.0 to 1.0)
# If match score is above this, we use the Dictionary answer.
# If below, we go to the LLM.
SIMILARITY_THRESHOLD = 0.70

# 2. Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_FILE = os.path.join(BASE_DIR, "data", "faq.json")

# 3. Model Configuration
# Embedding model for the dictionary (runs on CPU or GPU, very light)
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

# Generative Model for the fallback (Runs on GPU)
# You can change this to "mistralai/Mistral-7B-Instruct-v0.2" or "meta-llama/Meta-Llama-3-8B-Instruct"
LLM_MODEL_NAME = "microsoft/Phi-3-mini-4k-instruct" 

# GPU Settings
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🚀 Application config loaded. Running on: {DEVICE}")