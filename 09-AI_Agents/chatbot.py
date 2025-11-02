from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
import os
from dotenv import load_dotenv

# Get the directory of the current script to build absolute paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# Load environment variables from .env file in the script's directory
load_dotenv(os.path.join(SCRIPT_DIR, '.env'))

# Set your Google API key
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

# Constants with absolute paths
CHROMA_PERSIST_DIR = os.path.join(SCRIPT_DIR, "chroma_db")

def main():
    if not os.path.exists(CHROMA_PERSIST_DIR):
        print("Vector store not found. Please run `ingest.py` first to create it.")
        return

    print("Loading vector store...")
    # Use the same local embeddings model
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectordb = Chroma(persist_directory=CHROMA_PERSIST_DIR, embedding_function=embeddings)

    retriever = vectordb.as_retriever()
    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0)

    # Create a simple prompt template
    template = """Use the following pieces of context to answer the question at the end. 
If you don't know the answer, just say that you don't know, don't try to make up an answer.

Context: {context}

Question: {question}

Answer:"""
    
    prompt = PromptTemplate(template=template, input_variables=["context", "question"])
    
    # Create the chain
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)
    
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    print("Chatbot is ready. Type 'exit' to quit.")
    while True:
        query = input("You: ")
        if query.lower() == 'exit':
            break
        if query:
            try:
                result = rag_chain.invoke(query)
                print("Bot:", result)
            except Exception as e:
                print(f"Error: {e}")

if __name__ == "__main__":
    main()
