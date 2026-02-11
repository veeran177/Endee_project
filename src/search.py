from endee import Endee
from sentence_transformers import SentenceTransformer

# We initialize these outside the function to keep them in memory
model = SentenceTransformer('all-MiniLM-L6-v2')
client = Endee()

def perform_search(query_text, num_results=5):
    """Takes a string and returns the top matching products."""
    idx = client.get_index("ecommerce_products")
    
    # Convert text to vector
    query_vector = model.encode(query_text).tolist()
    
    # Search the database using the keyword signature we discovered
    results = idx.query(vector=query_vector, top_k=num_results)
    
    return results