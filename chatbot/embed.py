__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import chromadb
from sentence_transformers import SentenceTransformer
from chatbot.ingest import collect_documents

def run_ingestion():
    print("🤖 Initializing Vector Database...")
    client = chromadb.PersistentClient(path=".chroma")
    embed_model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
    
    try:
        client.delete_collection("erp_docs")
    except:
        pass
    collection = client.get_or_create_collection("erp_docs")

    print("📡 Fetching deep nested data from API...")
    raw_docs = collect_documents()
    
    if not raw_docs:
        print("❌ No data found to index.")
        return

    print(f"📥 Total items to process: {len(raw_docs)}")
    
    documents = [d["text"] for d in raw_docs]
    metadatas = [{"root_module": d["module"], "path": d["path"]} for d in raw_docs]
    ids = [f"id_{i}" for i in range(len(raw_docs))]

    print("🧠 Generating embeddings...")
    embeddings = embed_model.encode(documents, show_progress_bar=True).tolist()
    
    # --- BATCHING LOGIC START ---
    MAX_BATCH_SIZE = 5000  # Staying safely under the 5461 limit
    print(f"📦 Adding to database in batches of {MAX_BATCH_SIZE}...")
    
    for i in range(0, len(documents), MAX_BATCH_SIZE):
        batch_end = i + MAX_BATCH_SIZE
        collection.add(
            documents=documents[i:batch_end],
            metadatas=metadatas[i:batch_end],
            ids=ids[i:batch_end],
            embeddings=embeddings[i:batch_end]
        )
        print(f"✅ Indexed items {i} to {min(batch_end, len(documents))}")
    # --- BATCHING LOGIC END ---
    
    print(f"🚀 Success! {collection.count()} items are now ready in your ThinkPad's database.")

if __name__ == "__main__":
    run_ingestion()