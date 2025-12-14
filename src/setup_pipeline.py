import os
import glob
import json
from pypdf import PdfReader
from docx import Document
from llm_gateway import UnifiedLLMClient
from vector_store import VectorStore

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
        return text
    except Exception:
        return ""

def chunk_text(text, chunk_size=1000):
    """Simple chunker to split large text for Vector DB"""
    return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]

# --- MAIN PIPELINE ---
def run_pipeline():
    print("🚀 Starting Backend Setup Pipeline...")
    
    # 1. Init Components
    db = VectorStore()
    llm = UnifiedLLMClient()
    
    # 2. Ingestion: Files -> Vector DB
    target_folder = input("Enter path to raw documents folder: ").strip()
    files = glob.glob(os.path.join(target_folder, "*.*"))
    
    print(f"📂 Found {len(files)} files. Processing...")
    
    all_chunks = []
    for file in files:
        raw_text = extract_text_from_file(file)
        if raw_text:
            # Chunking is crucial for RAG
            chunks = chunk_text(raw_text)
            all_chunks.extend(chunks)
            print(f"   - Processed {os.path.basename(file)} ({len(chunks)} chunks)")

    if all_chunks:
        print("💾 Saving to Postgres Vector DB...")
        db.add_documents(all_chunks)
    else:
        print("⚠️ No content found to ingest.")

    # 3. Generate FAQ: Vector DB -> FAQ.json
    print("\n🧠 Generating FAQ from Database content...")
    full_knowledge = db.get_all_text()
    
    # Limit context for generation if too huge (optional, depends on model context window)
    full_knowledge = full_knowledge[:50000] 

    system_prompt = """
    Generate a FAQ JSON list based on the text provided.
    Format: [{"questions": ["..."], "answer": "..."}]
    Cover the most important 90% of facts.
    """
    
    response = llm.generate_text(
        system_prompt, 
        f"Content: {full_knowledge}", 
        json_mode=True
    )
    
    try:
        # Sanitize and save
        clean_json = response.replace("```json", "").replace("```", "").strip()
        faq_data = json.loads(clean_json)
        
        with open("data/faq.json", "w", encoding="utf-8") as f:
            json.dump(faq_data, f, indent=4)
        print("✅ faq.json created successfully!")
        
    except Exception as e:
        print(f"❌ Error creating FAQ: {e}")

if __name__ == "__main__":
    run_pipeline()