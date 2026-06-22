import os
import chromadb
from sentence_transformers import SentenceTransformer

# Resolve DB path relative to the root of the project (parent of assistant folder)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(BASE_DIR, "chroma_db")

# Initialize model and database client lazily to avoid heavy loading on import if needed,
# or initialize them globally as per the skeleton.
_model = None
_client = None
_collection = None

def get_resources():
    global _model, _client, _collection
    if _model is None:
        _model = SentenceTransformer("all-MiniLM-L6-v2")
    if _client is None:
        _client = chromadb.PersistentClient(path=DB_PATH)
    if _collection is None:
        try:
            _collection = _client.get_collection("catalogue")
        except Exception as e:
            # If the database doesn't exist yet, we will raise a helpful message
            raise RuntimeError(f"ChromaDB collection 'catalogue' could not be loaded at {DB_PATH}. "
                               f"Please run assistant/indexer.py first to build the index. Error: {e}")
    return _model, _collection

def search(query: str, top_k: int = 5) -> list[dict]:
    model, collection = get_resources()
    embedding = model.encode([query]).tolist()
    results = collection.query(query_embeddings=embedding, n_results=top_k)
    if results and "metadatas" in results and results["metadatas"]:
        return results["metadatas"][0]  # list of metadata dicts
    return []

if __name__ == "__main__":
    # Small test when run directly
    try:
        results = search("brake pads for Pulsar 150")
        print("Search works! Top results:")
        for r in results:
            print(f"- {r['sku']}: {r['name']} | ₹{r['price_inr']} | Stock: {r['stock']}")
    except Exception as e:
        print(f"Test search failed: {e}")
