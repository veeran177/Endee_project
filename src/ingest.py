import pandas as pd
import numpy as np
from endee import Endee
from sentence_transformers import SentenceTransformer

def run_ingestion(csv_path=r'D:\Endee\endee_project\data\product_description.csv'):
    # 1. Load the data
    print("Loading data...")
    df = pd.read_csv(csv_path)
    
    # 2. Initialize Model and Endee
    model = SentenceTransformer('all-MiniLM-L6-v2')
    client = Endee()
    
    index_name = "ecommerce_products"
    
    # 3. Create Index (if it doesn't exist)
    try:
        client.create_index(name=index_name, dimension=384, space_type="cosine")
        print(f"✅ Index '{index_name}' created.")
    except Exception as e:
        print(f"Index status: {e}")

    idx = client.get_index(index_name)

    # 4. Prepare and Upload Data
    print("Vectorizing and uploading products...")
    input_array = []
    
    for i, row in df.iterrows():
        # Combine title and description for better search context
        text_to_vectorize = f"{row['title']} {row['description']}"
        vector = model.encode(text_to_vectorize).tolist()
        
        input_array.append({
            "id": str(row['id']),
            "vector": vector,
            "metadata": {
                "title": str(row['title']),
                "category": str(row['category']),
                "price": float(row['price'])
            }
        })

    # Batch upload
    idx.upsert(input_array)
    print(f"🚀 Success! {len(input_array)} products are now in the database.")

if __name__ == "__main__":
    run_ingestion()