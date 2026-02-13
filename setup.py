import os
import json
import fitz
from ccore.ingestion import extract_text_from_page
from ccore.indexing import create_embeddings, save_to_faiss

def main():
    # Ensure directories exist
    if not os.path.exists("vector_db"):
        os.makedirs("vector_db")
    if not os.path.exists("data"):
        os.makedirs("data")

    # 1. Ingest Data
    # Find any PDF in the data folder
    pdf_files = [f for f in os.listdir("data") if f.lower().endswith('.pdf')]
    
    if not pdf_files:
        print("No PDF found. Generating a sample PDF for demonstration...")
        pdf_path = os.path.join("data", "sample.pdf")
        doc = fitz.open()
        page = doc.new_page()
        page.insert_text((50, 50), "Welcome to your RAG system!\nThis is a generated sample PDF.\nPlease add your own PDF to the data folder to query your own documents.", fontsize=12)
        doc.save(pdf_path)
    else:
        pdf_path = os.path.join("data", pdf_files[0])

    print(f"Extracting content from {pdf_path}...")
    doc = fitz.open(pdf_path)
    chunks = []
    for page in doc:
        text, _ = extract_text_from_page(page)
        chunks.append(text)
    
    # Save chunks to JSON so app.py can read the text later
    with open("data/chunks.json", "w") as f:
        json.dump(chunks, f)

    # 2. Create Embeddings & Save Index
    print("Generating embeddings (this may take a moment)...")
    embeddings = create_embeddings(chunks)
    
    print("Saving FAISS index...")
    save_to_faiss(embeddings, metadata=None)
    print("Setup complete! You can now run 'streamlit run app.py'")

if __name__ == "__main__":
    main()