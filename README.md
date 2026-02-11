# Semantic Product Search Engine
An end-to-end AI application that understands user intent to find products, built for the Endee Internship.

<img width="1920" height="1080" alt="Screenshot (38)" src="https://github.com/user-attachments/assets/80c15282-0eb2-4cc4-99d0-6d6490551789" />


## How it Works
Unlike traditional keyword search, this system uses **Vector Embeddings**. 
- **Model**: `all-MiniLM-L6-v2` (Sentence-Transformers)
- **Database**: **Endee** (Vector Database)
- **UI**: **Streamlit**

## Setup Instructions
1. **Start the Database**:
   ```bash
   docker run -p 8080:8080 endee/server

2. **Install Dependencies:**
pip install -r requirements.txt

4. **Ingest Data:**
python src/ingest.py

4. **Launch App:**
streamlit run app.py

**What I Learned**
Implementing vector similarity search with Cosine distance.
Handling real-world data ingestion and batch upserts.
Building a modular Python project structure (separation of concerns).

