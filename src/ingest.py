"""
ingest.py
Modular RAG ingestion pipeline for census data.
Loads, chunks, embeds, and stores documents in a FAISS vectorstore.
"""

from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from modules import load_pdf, chunk_documents

def ingest_pdf(pdf_path):
    print("Starting ingestion pipeline...")

    # Step 1: Load the PDF
    documents = load_pdf(pdf_path)
    print(f"Loaded {len(documents)} documents")

    # Step 2: Chunk the text
    chunks = chunk_documents(documents)
    print(f"Chunked into {len(chunks)} passages")

    # Step 3: Embed and store
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local("faiss_index")

    print("Vectorstore built successfully\nIngestion complete.")

if __name__ == "__main__":
    ingest_pdf("data/Census_2022_Municipal_factsheet-Web.pdf")
