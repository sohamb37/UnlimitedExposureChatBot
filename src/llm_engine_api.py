import psycopg2
from openai import OpenAI
from config import OPENAI_API_KEY, LLM_MODEL_NAME, DB_CONNECTION_STRING

# Use a chat model like 'gpt-4o' or 'gpt-3.5-turbo'
# Make sure LLM_MODEL_NAME in config.py is set to one of these
class RAGEngine:
    def __init__(self):
        print(f"Initializing RAG Engine with model: {LLM_MODEL_NAME}...")
        
        try:
            # 1. Initialize OpenAI Client
            self.client = OpenAI(api_key=OPENAI_API_KEY)
            
            # 2. Connect to Postgres (Simulated connection based on your request)
            self.conn = psycopg2.connect(DB_CONNECTION_STRING)
            
            print("RAG Engine (OpenAI + PGVector) ready.")
            
        except Exception as e:
            print(f"❌ Error initializing RAG Engine: {e}")
            self.client = None

    def get_query_embedding(self, text):
        """Generates embedding for the user's query to match against DB."""
        text = text.replace("\n", " ")
        return self.client.embeddings.create(
            input=[text], 
            model="text-embedding-3-small"
        ).data[0].embedding

    def retrieve_context(self, query_embedding, top_k=3):
        """
        Retrieves the most relevant text chunks from Postgres.
        ASSUMPTION: You have a table named 'knowledge_base' with 'content' and 'embedding'.
        """
        context_chunks = []
        
        # SQL to find nearest neighbors using Cosine Distance (<=>)
        sql = """
            SELECT content 
            FROM knowledge_base 
            ORDER BY embedding <=> %s::vector 
            LIMIT %s;
        """
        
        try:
            with self.conn.cursor() as cur:
                cur.execute(sql, (query_embedding, top_k))
                rows = cur.fetchall()
                # Extract just the text content from the rows
                context_chunks = [row[0] for row in rows]
        except Exception as e:
            print(f"⚠️ Database retrieval failed: {e}")
            
        return context_chunks

    def generate(self, user_query):
        """
        Performs the full RAG pipeline: 
        Embed Query -> Retrieve Context -> Generate Answer
        """
        if not self.client:
            return "Error: OpenAI client is not initialized."

        # Step 1: Embed the user's question
        query_vector = self.get_query_embedding(user_query)
        
        # Step 2: Retrieve ranked items from PG Database
        # We get the top 3 most relevant chunks
        retrieved_items = self.retrieve_context(query_vector, top_k=3)
        
        if not retrieved_items:
            # Fallback if DB is empty or fails
            print("No context retrieved. Answering based on general knowledge.")
            context_str = "No specific context available."
        else:
            # Join the chunks into a single string
            context_str = "\n\n---\n\n".join(retrieved_items)

        # Step 3: Construct the RAG Prompt
        # We give the LLM the retrieved text and the user's question
        messages = [
            {
                "role": "system", 
                "content": (
                    "You are a helpful assistant for a restaurant. "
                    "Use the following pieces of retrieved context to answer the user's question. "
                    "If the answer is not in the context, strictly say 'I do not have that information'."
                )
            },
            {
                "role": "user", 
                "content": f"Context:\n{context_str}\n\nQuestion: {user_query}"
            }
        ]
        
        # Step 4: Generate Response via OpenAI
        try:
            response = self.client.chat.completions.create(
                model=LLM_MODEL_NAME, # e.g., "gpt-4o"
                messages=messages,
                temperature=0.3, # Keep it low for factual grounding
                max_tokens=512
            )
            return response.choices[0].message.content
            
        except Exception as e:
            return f"Error generating response: {e}"

# --- usage example ---
if __name__ == "__main__":
    rag = RAGEngine()
    answer = rag.generate("What is the return policy?")
    print(answer)