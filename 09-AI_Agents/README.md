
# RAG-based Chatbot (with Google Generative AI)

This project implements a Retrieval-Augmented Generation (RAG) chatbot using LangChain, Google Generative AI, and ChromaDB. The chatbot can answer questions about a collection of documents provided by the user.

## Features

- 🤖 RAG-based Q&A system using Google's Gemini 2.0 Flash model
- 📚 Vector store using ChromaDB for efficient document retrieval
- 🔍 Local embeddings using HuggingFace's `all-MiniLM-L6-v2` model
- 💬 Interactive command-line chat interface

## Setup

1.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

2.  **Configure API Key:**
    - Copy `.env.example` to `.env`:
      ```bash
      cp .env.example .env
      ```
    - Get a Google API Key from [Google AI Studio](https://aistudio.google.com/app/apikey)
    - Add your API key to the `.env` file:
      ```
      GOOGLE_API_KEY=your_actual_api_key_here
      ```

3.  **Add your documents:**
    - Place text files you want to chat with into the `documents/` directory
    - A sample document is already included

4.  **Ingest documents:**
    - Run the ingestion script to process documents and create the vector store:
      ```bash
      python ingest_simple.py
      ```
    - Note: Use `ingest_simple.py` for faster processing, or `ingest.py` for more advanced text splitting

5.  **Run the chatbot:**
    ```bash
    python chatbot.py
    ```

## Usage

Once the chatbot is running, you can ask questions about your documents:

```
You: What is Retrieval-Augmented Generation?
Bot: Retrieval-Augmented Generation is a technique for enhancing the accuracy and reliability of large language models with facts fetched from external sources.

You: exit
```

Type `exit` to quit the chatbot.

## Project Structure

```
09-AI_Agents/
├── chatbot.py           # Main chatbot application
├── ingest.py            # Advanced document ingestion script
├── ingest_simple.py     # Simple document ingestion script (recommended)
├── requirements.txt     # Python dependencies
├── .env.example         # Example environment variables
├── .env                 # Your API keys (not tracked by git)
├── documents/           # Your documents go here
│   └── sample.txt
└── chroma_db/           # Vector database (auto-generated, not tracked by git)
```

## Requirements

- Python 3.8+
- Google API key for Gemini models
- Internet connection for downloading embeddings model (first run only)

## Notes

- The vector database (`chroma_db/`) is automatically created and should not be committed to git
- Your `.env` file containing API keys is excluded from git for security
- First run may take longer as it downloads the embedding model
