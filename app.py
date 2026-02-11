import streamlit as st
from endee import Endee
from sentence_transformers import SentenceTransformer
import numpy as np

# --- 1. Page Configuration ---
st.set_page_config(page_title="AI Product Search", page_icon="🛍️")
st.title("🛍️ Smart E-Commerce Search")
st.markdown("Search for products using natural language (e.g., *'something for my skin'*)")

# --- 2. Load Resources (Cached) ---
@st.cache_resource
def load_all():
    # Load the embedding model
    model = SentenceTransformer('all-MiniLM-L6-v2')
    # Connect to Endee
    client = Endee() 
    index = client.get_index("ecommerce_products")
    return model, index

try:
    model, idx = load_all()
    
    # --- 3. UI Sidebar/Search ---
    query_text = st.text_input("What are you looking for?", placeholder="Enter keywords or descriptions...")

    if query_text:
        with st.spinner("Analyzing intent..."):
            # Convert query to vector
            query_vector = model.encode(query_text).tolist()
            
            # Query the Endee index
            # Using the signature we discovered: (vector, top_k, ...)
            results = idx.query(vector=query_vector, top_k=6)

            if not results:
                st.warning("No products found. Try a different search!")
            else:
                # --- 4. Display Results in a Grid ---
                cols = st.columns(2)
                for i, res in enumerate(results):
                    # Robust metadata extraction based on our debugging
                    meta = res.metadata if hasattr(res, 'metadata') else res.get('metadata', {})
                    
                    with cols[i % 2]:
                        with st.container(border=True):
                            st.subheader(meta.get('title', 'Unknown Product'))
                            st.write(f"**Category:** {meta.get('category', 'N/A')}")
                            st.write(f"**Price:** ${meta.get('price', '0.00')}")
                            # Optional: Short description if available
                            if 'description' in meta:
                                st.caption(meta['description'][:100] + "...")

except Exception as e:
    st.error("⚠️ Connection Error")
    st.info("Make sure your Endee server is running in Docker (port 8080) and that you have already run the indexing script in your notebook.")
    st.expander("Show Technical Error").write(e)