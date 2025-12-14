import psycopg2
from psycopg2.extras import execute_values
from config import settings
from llm_gateway import UnifiedLLMClient

class VectorStore:
    def __init__(self):
        self.conn = psycopg2.connect(
            host=settings.DB_HOST,
            port=settings.DB_PORT,
            dbname=settings.DB_NAME,
            user=settings.DB_USER,
            password=settings.DB_PASSWORD
        )
        self.client = UnifiedLLMClient()
        self._init_db()

    def _init_db(self):
        """Enable pgvector extension and create table."""
        with self.conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
            cur.execute("""
                CREATE TABLE IF NOT EXISTS documents (
                    id SERIAL PRIMARY KEY,
                    content TEXT,
                    embedding vector(1536) 
                );
            """)
            # Note: 1536 is the dimension for OpenAI embeddings. 
            # Adjust if using a different model (e.g., 1024 for Mistral).
        self.conn.commit()

    def add_documents(self, texts: list[str]):
        """Generate embeddings and save to DB."""
        if not texts: return
        
        data = []
        for text in texts:
            vector = self.client.get_embedding(text)
            data.append((text, vector))

        with self.conn.cursor() as cur:
            execute_values(cur, 
                "INSERT INTO documents (content, embedding) VALUES %s", 
                data
            )
        self.conn.commit()
        print(f"✅ Added {len(texts)} documents to Vector DB.")

    def search(self, query: str, limit: int = 3):
        """Semantic search using Cosine Similarity."""
        query_vector = self.client.get_embedding(query)
        
        with self.conn.cursor() as cur:
            cur.execute("""
                SELECT content, 1 - (embedding <=> %s::vector) as similarity
                FROM documents
                ORDER BY similarity DESC
                LIMIT %s;
            """, (query_vector, limit))
            results = cur.fetchall()
            
        return [row[0] for row in results]

    def get_all_text(self):
        """Fetch all text for FAQ generation."""
        with self.conn.cursor() as cur:
            cur.execute("SELECT content FROM documents;")
            return " ".join([row[0] for row in cur.fetchall()])