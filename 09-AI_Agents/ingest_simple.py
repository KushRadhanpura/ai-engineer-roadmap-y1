from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
import os
import shutil
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

def main():
    # Check if the chroma_db directory exists and delete it to start fresh
    if os.path.exists(CHROMA_PERSIST_DIR):
        print("Existing ChromaDB found. Deleting to start fresh.")
        shutil.rmtree(CHROMA_PERSIST_DIR)

    print("Loading documents...")
    
    # Read the sample.txt file directly
    sample_file = os.path.join(DOCUMENTS_DIR, "sample.txt")
    if not os.path.exists(sample_file):
        print("sample.txt not found!")
        return
    
    with open(sample_file, 'r') as f:
        content = f.read()
    
    # Create a simple document
    doc = Document(page_content=content, metadata={"source": "sample.txt"})
    
    # Split into chunks manually
    chunks = []
    lines = content.split('\n')
    for i, line in enumerate(lines):
        if line.strip():  # Only add non-empty lines
            chunks.append(Document(
                page_content=line,
                metadata={"source": "sample.txt", "line": i}
            ))
    
    print(f"Created {len(chunks)} chunks.")
    
    print("Creating embeddings and storing in ChromaDB...")
    # Use a local model for embeddings to avoid API calls
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    # Create ChromaDB from all texts at once
    vectordb = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_PERSIST_DIR
    )
    
    print(f"Successfully created vector store with {len(chunks)} chunks!")
    print("You can now run chatbot.py to ask questions.")

if __name__ == "__main__":
    main()
