# Shakespeare Insights: AI-Powered Q&A

This project combines the power of large language models (LLMs) and retrieval-based techniques to provide users with accurate and informative answers to their queries. The RAG system is particularly optimized for the Complete Works of William Shakespeare, allowing users to explore various works such as Hamlet, Romeo and Juliet, Macbeth, The Tempest, The Sonnets, and many others.

## Features
- Document processing and chunking
- FAISS vector store for efficient similarity search
- Real-time Question Answering
- Streamlit web interface
- Conversation history
- Source document tracking
- Docker support

## Technologies used

- LangChain -  A framework for building applications with large language models and integrating retrieval capabilities, allowing for the combination of LLMs with document search.

- Hugging Face Transformers - Leveraging pre-trained models like Flan-T5 for answer generation and embeddings for semantic search.

- FAISS (Facebook AI Similarity Search) - Efficient vector search engine that powers the document retrieval by quickly identifying the most relevant document chunks.

- Streamlit - The front-end framework that allows users to interact with the system via a simple and interactive web interface.

- Docker - Full support for containerization using Docker, ensuring that the entire system can be easily deployed, including the RAG system, FAISS vector store, and web interface.



## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Add documents:
- Place your .txt files in the `data/` directory

3. Run locally:
```bash
streamlit run app/app.py
```

Or with Docker:
```bash
docker-compose up --build
```
