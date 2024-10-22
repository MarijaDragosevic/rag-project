import os
from typing import List, Dict
from dataclasses import dataclass
from pathlib import Path

import torch
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_huggingface import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, T5ForConditionalGeneration

@dataclass
class DocumentChunk:
    content: str
    metadata: Dict

class DocumentProcessor:
    def __init__(self, chunk_size: int = 400, chunk_overlap: int = 150):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            is_separator_regex=False,
        )

    def load_documents(self, directory_path: str) -> List:
        documents = []
        directory = Path(directory_path)
        
        if not directory.exists():
            raise FileNotFoundError(f"Directory not found: {directory}")
        
        text_files = list(directory.glob("*.txt"))
        if not text_files:
            raise ValueError(f"No .txt files found in {directory}")
            
        for file_path in text_files:
            try:
                loader = TextLoader(str(file_path))
                docs = loader.load()
                chunks = self.text_splitter.split_documents(docs)
                documents.extend(chunks)
            except Exception as e:
                print(f"Error loading {file_path}: {e}")
        
        if not documents:
            raise ValueError("No documents were successfully loaded")
            
        return documents

class EmbeddingsManager:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.embeddings = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={'device': device}
        )
        self.vector_store = None

    def create_vector_store(self, documents: List, store_path: str = "vector_store"):
        if not documents:
            raise ValueError("No documents provided to create vector store")
        
        
        os.makedirs(store_path, exist_ok=True)
        
        try:
            self.vector_store = FAISS.from_documents(documents, self.embeddings)
            self.vector_store.save_local(store_path)
        except Exception as e:
            raise Exception(f"Failed to create vector store: {str(e)}")

    def load_vector_store(self, store_path: str = "vector_store"):
        index_path = Path(store_path) / "index.faiss"
        pkl_path = Path(store_path) / "index.pkl"
        
        if not index_path.exists() or not pkl_path.exists():
            raise FileNotFoundError(f"Vector store files not found at {store_path}")
            
        try:
            self.vector_store = FAISS.load_local(store_path, self.embeddings, allow_dangerous_deserialization=True)
        except Exception as e:
            raise Exception(f"Failed to load vector store: {str(e)}")

class RAGEngine:
    def __init__(self, vector_store):
        if vector_store is None:
            raise ValueError("Vector store cannot be None")
            
        self.vector_store = vector_store
        self.model_name =  "google/flan-t5-large" 
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = T5ForConditionalGeneration.from_pretrained(self.model_name)
            
            device = 0 if torch.cuda.is_available() else -1
            
            self.pipe = pipeline(
                "text2text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                max_length=512,
                device=device
            )
            
            self.llm = HuggingFacePipeline(pipeline=self.pipe)
            
            self.qa_chain = RetrievalQA.from_chain_type(
                llm=self.llm,
                chain_type="refine",
                retriever=self.vector_store.as_retriever(search_kwargs={"k": 5}),
                return_source_documents=True
            )
        except Exception as e:
            raise Exception(f"Failed to initialize RAG engine: {str(e)}")

    def query(self, question: str) -> Dict:
        if not question or not question.strip():
            raise ValueError("Question cannot be empty")
            
        try:
            result = self.qa_chain.invoke({"query": question})
            return {
                "answer": result["result"],
                "source_documents": result["source_documents"]
            }
        except Exception as e:
            raise Exception(f"Error processing query: {str(e)}")

def setup_directories():
    directories = ["data", "vector_store"]
    for directory in directories:
        os.makedirs(directory, exist_ok=True)