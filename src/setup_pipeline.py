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
from llm_gateway import UnifiedLLMClient
from vector_store import VectorStore
from webscraper import WebScraper

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

    # Limit context for generation to avoid huge costs/errors
    # 100k chars is approx 25k tokens. 
    context_slice = full_knowledge[:100000] 

    print("🧠 Generating FAQ.json via LLM (this may take a moment)...")
    system_prompt = """
    Generate a FAQ JSON list based on the text provided.
    Format: [{"questions": ["..."], "answer": "..."}]
    Create comprehensive answers.
    """
    
    response = llm.generate_text(
        system_prompt, 
        f"Content Source:\n{context_slice}", 
        json_mode=True
    )
    
    try:
        clean_json = response.replace("```json", "").replace("```", "").strip()
        faq_data = json.loads(clean_json)
        
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