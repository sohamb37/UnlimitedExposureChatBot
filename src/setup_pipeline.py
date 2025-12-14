import os
import glob
import json
import sys

# Add parent directory to path so we can import config.py from root
# We use insert(0, ...) to prioritize this path over standard library paths
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from config import settings
from pypdf import PdfReader
from docx import Document
from src.llm_gateway import UnifiedLLMClient
from src.vector_store import VectorStore
from src.webscraper import WebScraper

# --- HELPER: Read Files ---
def extract_text_from_file(filepath):
    ext = os.path.splitext(filepath)[1].lower()
    text = ""
    try:
        if ext in ['.txt', '.md']:
            with open(filepath, 'r', encoding='utf-8') as f: text = f.read()
        elif ext == '.pdf':
            reader = PdfReader(filepath)
            for page in reader.pages: text += page.extract_text() + "\n"
        elif ext == '.docx':
            doc = Document(filepath)
            for para in doc.paragraphs: text += para.text + "\n"
        return f"\n--- SOURCE FILE: {os.path.basename(filepath)} ---\n{text}"
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        return ""

def chunk_text(text, chunk_size=2000):
    """
    Splits text into chunks. 
    Larger chunks (2000 chars) are better for RAG context.
    """
    if not text: return []
    return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]

# --- MAIN PIPELINE ---
def run_pipeline():
    print("🚀 Starting Backend Setup Pipeline...")
    
    # 1. Init Components
    try:
        db = VectorStore()
        llm = UnifiedLLMClient()
        scraper = WebScraper()
    except Exception as e:
        print(f"❌ Initialization Failed: {e}")
        return
    
    all_chunks = []

    # --- PHASE 1: FILE INGESTION ---
    target_folder = input("Enter path to raw documents folder (or press Enter to skip): ").strip()
    
    if target_folder and os.path.exists(target_folder):
        files = glob.glob(os.path.join(target_folder, "*.*"))
        print(f"📂 Found {len(files)} files.")
        for file in files:
            raw_text = extract_text_from_file(file)
            chunks = chunk_text(raw_text)
            all_chunks.extend(chunks)
            print(f"   - Processed file: {os.path.basename(file)} ({len(chunks)} chunks)")
    elif target_folder:
        print(f"⚠️ Folder not found: {target_folder}")

    # --- PHASE 2: WEB SCRAPING ---
    scrape_input = input("Enter website URL to scrape (or press Enter to skip): ").strip()
    
    if scrape_input:
        # The scraper just returns text. It does NOT save to DB.
        web_text = scraper.scrape_page(scrape_input)
        
        if web_text:
            web_chunks = chunk_text(web_text)
            all_chunks.extend(web_chunks)
            print(f"   - Processed website: {scrape_input} ({len(web_chunks)} chunks)")
        else:
            print("   ⚠️ Scraper returned no content.")

    # --- PHASE 3: SAVE TO VECTOR DB ---
    # This is where the actual saving happens for BOTH files and web data
    if all_chunks:
        print(f"\n💾 Saving {len(all_chunks)} total chunks to Postgres Vector DB...")
        db.add_documents(all_chunks)
        print("✅ Data successfully stored in Vector DB.")
    else:
        print("⚠️ No content found to ingest. Skipping DB save.")

    # --- PHASE 4: GENERATE FAQ ---
    print("\n🧠 Reading full knowledge base from Database to generate FAQ...")
    full_knowledge = db.get_all_text()
    
    if not full_knowledge:
        print("❌ Database is empty. Cannot generate FAQ.")
        return

    # Optimization: Increased context window significantly (approx 112k tokens).
    # Since this is a one-time setup cost, we prioritize comprehensive coverage
    # to reduce recurring RAG costs in production.
    context_slice = full_knowledge[:450000]

    print("🧠 Generating FAQ.json via LLM (this may take a moment)...")
    system_prompt = """
    You are an expert customer support architect. Your goal is to create a robust FAQ database from the provided text.

    CRITICAL INSTRUCTIONS:
    1. **Exhaustive Extraction:** Analyze the text deeply. Create a Q&A pair for every distinct fact, service, price, policy, or feature found. Do not summarize the whole document into one answer; break it down.
    2. **Question Diversity:** For every Answer, generate 5 to 10 question variations. Mix short keywords (e.g., "Pricing"), natural questions ("How much is it?"), and specific scenarios ("Cost for enterprise plan?"). This is vital for semantic matching.
    3. **Answer Quality:** Answers must be polite, unambiguous, and self-contained. Never reference "the text" or "above/below". Assume the user only sees this one answer.
    4. **Output Format:** Return valid JSON. Wrap the list of FAQs in a root object with the key "faqs".
    
    Structure:
    {
        "faqs": [
            {
                "questions": ["variant 1", "variant 2", "variant 3", ...],
                "answer": "The specific, polite answer."
            }
        ]
    }
    """
    
    response = llm.generate_text(
        system_prompt, 
        f"Content Source:\n{context_slice}",
        temperature=0.1, # Low temperature for factual/structural consistency
        json_mode=True
    )
    
    try:
        clean_json = response.replace("```json", "").replace("```", "").strip()
        json_output = json.loads(clean_json)
        
        # Robust Parsing: Handle both wrapped object (preferred) and direct list (fallback)
        if isinstance(json_output, dict) and "faqs" in json_output:
            faq_data = json_output["faqs"]
        elif isinstance(json_output, list):
            faq_data = json_output
        else:
            # Last resort if the key isn't "faqs" but it's a dictionary
            print("⚠️ Warning: JSON structure unexpected, attempting to extract values...")
            faq_data = list(json_output.values())[0] if json_output else []

        # Calculate path to data folder (root/data)
        data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
        os.makedirs(data_dir, exist_ok=True)
        output_path = os.path.join(data_dir, "faq.json")
        
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(faq_data, f, indent=4)
        print(f"✅ faq.json created successfully at {output_path}!")
        
    except Exception as e:
        print(f"❌ Error creating FAQ JSON: {e}")
        print("Raw Output was:", response)

if __name__ == "__main__":
    run_pipeline()