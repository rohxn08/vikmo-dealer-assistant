import os
import pandas as pd
import chromadb
from sentence_transformers import SentenceTransformer

def build_index(catalogue_path="catalogue.csv", db_path="./chroma_db"):
    if not os.path.exists(catalogue_path):
        # Try parent directory or relative to this script
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        alt_path = os.path.join(base_dir, catalogue_path)
        if os.path.exists(alt_path):
            catalogue_path = alt_path
        else:
            raise FileNotFoundError(f"Catalogue file not found at {catalogue_path} or {alt_path}")

    print(f"Loading catalogue from {catalogue_path}...")
    df = pd.read_csv(catalogue_path)
    
    print("Loading embedding model (all-MiniLM-L6-v2)...")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    
    print(f"Initializing ChromaDB client at {db_path}...")
    client = chromadb.PersistentClient(path=db_path)
    
    # Reset/recreate the collection to avoid duplicates or issues if run multiple times
    try:
        client.delete_collection("catalogue")
        print("Existing collection deleted.")
    except Exception:
        pass
        
    collection = client.get_or_create_collection("catalogue")

    docs, ids, metadatas = [], [], []
    for _, row in df.iterrows():
        # Handle nan fields if any
        name = str(row['name']) if pd.notna(row['name']) else ""
        category = str(row['category']) if pd.notna(row['category']) else ""
        fitment = str(row['vehicle_fitment']) if pd.notna(row['vehicle_fitment']) else ""
        brand = str(row['brand']) if pd.notna(row['brand']) else ""
        description = str(row['description']) if pd.notna(row['description']) else ""
        
        text = f"{name}. Category: {category}. Fits: {fitment}. Brand: {brand}. {description}"
        docs.append(text)
        ids.append(str(row['sku']))
        metadatas.append({
            "sku": str(row['sku']),
            "name": name,
            "price_inr": int(row['price_inr']) if pd.notna(row['price_inr']) else 0,
            "stock": int(row['stock']) if pd.notna(row['stock']) else 0,
            "vehicle_fitment": fitment,
            "category": category,
            "brand": brand
        })

    print("Encoding documents...")
    embeddings = model.encode(docs, show_progress_bar=True).tolist()
    
    print("Adding to ChromaDB collection...")
    collection.add(documents=docs, embeddings=embeddings, ids=ids, metadatas=metadatas)
    print(f"Successfully indexed {len(ids)} SKUs.")

if __name__ == "__main__":
    # If run directly, run build_index in the default configuration
    build_index()
