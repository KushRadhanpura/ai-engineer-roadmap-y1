from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader
from langchain_huggingface import HuggingFaceEmbeddings
import os
import time
from dotenv import load_dotenv

# Get the directory of the current script to build absolute paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# Load environment variables from .env file in the script's directory
load_dotenv(os.path.join(SCRIPT_DIR, '.env'))

# Set your Google API key
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

# Constants with absolute paths
DOCUMENTS_DIR = os.path.join(SCRIPT_DIR, "documents")
CHROMA_PERSIST_DIR = os.path.join(SCRIPT_DIR, "chroma_db")
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
BATCH_SIZE = 15  # Process 15 chunks at a time
DELAY_SECONDS = 90 # Wait for 90 seconds between batches

def main():
    # Check if the chroma_db directory exists and delete it to start fresh
    if os.path.exists(CHROMA_PERSIST_DIR):
        print("Existing ChromaDB found. Deleting to start fresh.")
        import shutil
        shutil.rmtree(CHROMA_PERSIST_DIR)

    print("Loading documents...")
    loader = DirectoryLoader(DOCUMENTS_DIR)
    documents = loader.load()
    if not documents:
        print("No documents found in the 'documents' directory. Please add a file to process.")
        return

    print(f"Loaded {len(documents)} documents.")

    print("Splitting documents into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    texts = text_splitter.split_documents(documents)
    print(f"Split into {len(texts)} chunks.")

    print("Creating embeddings and storing in ChromaDB...")
    # Use a local model for embeddings to avoid API calls
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    # Create ChromaDB from all texts at once
    vectordb = Chroma.from_documents(
        documents=texts,
        embedding=embeddings,
        persist_directory=CHROMA_PERSIST_DIR
    )
    
    vectordb.persist()
    print("Ingestion complete. Vector store created in 'chroma_db'.")

if __name__ == "__main__":
    main()
