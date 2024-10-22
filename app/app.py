import streamlit as st
from typing import Optional
import time
from pathlib import Path
import os

from main import DocumentProcessor, EmbeddingsManager, RAGEngine

works = [
    "The Sonnets",
    "All’s Well That Ends Well",
    "The Tragedy of Antony and Cleopatra",
    "As You Like It",
    "The Comedy of Errors",
    "The Tragedy of Coriolanus",
    "Cymbeline",
    "The Tragedy of Hamlet, Prince of Denmark",
    "The First Part of King Henry the Fourth",
    "The Second Part of King Henry the Fourth",
    "The Life of King Henry the Fifth",
    "The First Part of Henry the Sixth",
    "The Second Part of King Henry the Sixth",
    "The Third Part of King Henry the Sixth",
    "King Henry the Eighth",
    "The Life and Death of King John",
    "The Tragedy of Julius Caesar",
    "The Tragedy of King Lear",
    "Love’s Labour’s Lost",
    "The Tragedy of Macbeth",
    "Measure for Measure",
    "The Merchant of Venice",
    "The Merry Wives of Windsor",
    "A Midsummer Night’s Dream",
    "Much Ado About Nothing",
    "The Tragedy of Othello, the Moor of Venice",
    "Pericles, Prince of Tyre",
    "King Richard the Second",
    "King Richard the Third",
    "The Tragedy of Romeo and Juliet",
    "The Taming of the Shrew",
    "The Tempest",
    "The Life of Timon of Athens",
    "The Tragedy of Titus Andronicus",
    "Troilus and Cressida",
    "Twelfth Night; Or, What You Will",
    "The Two Gentlemen of Verona",
    "The Two Noble Kinsmen",
    "The Winter’s Tale",
    "A Lover’s Complaint",
    "The Passionate Pilgrim",
    "The Phoenix and the Turtle",
    "The Rape of Lucrece",
    "Venus and Adonis"
]


suggested_questions = [
    "Who wrote the Complete Works of William Shakespeare?",
    "Does King John die?",
    "Who is Romeo in love with?",
    "Who wrote The Sonnets?",
    "Who does Hamlet love?"
]

class RAGWebInterface:
    def __init__(self):
        self.initialize_session_state()
        self.setup_rag_system()

    @staticmethod
    def initialize_session_state():
        if 'rag_engine' not in st.session_state:
            st.session_state.rag_engine = None
        if 'history' not in st.session_state:
            st.session_state.history = []
        if 'selected_question' not in st.session_state:
            st.session_state.selected_question = ""

    def setup_rag_system(self):
        try:
            
            os.makedirs("vector_store", exist_ok=True)
            
            embeddings_manager = EmbeddingsManager()
            vector_store_path = Path("vector_store")
            index_path = vector_store_path / "index.faiss"

            # Check if both index.faiss and index.pkl exist
            if not index_path.exists() or not (vector_store_path / "index.pkl").exists():
                st.warning("Vector store not found. Creating a new one...")
                doc_processor = DocumentProcessor()
                
                
                data_path = Path("data")
                if not data_path.exists() or not list(data_path.glob("*.txt")):
                    st.error("No text documents found in the data directory. Please add some .txt files to the 'data' folder.")
                    return
                
                with st.spinner("Processing documents..."):
                    documents = doc_processor.load_documents(str(data_path))
                    if not documents:
                        st.error("No documents were successfully loaded. Please check your text files.")
                        return
                    
                    st.info(f"Loaded {len(documents)} document chunks. Creating vector store...")
                    embeddings_manager.create_vector_store(documents, str(vector_store_path))
                st.success("✅ Created new vector store")
            else:
                # Load the existing vector store
                embeddings_manager.load_vector_store(str(vector_store_path))
                st.success("✅ Loaded existing vector store")
            
            if st.session_state.rag_engine is None:
                st.session_state.rag_engine = RAGEngine(embeddings_manager.vector_store)
                
        except Exception as e:
            st.error(f"Error setting up RAG system: {str(e)}")
            st.error("Please ensure you have:")
            st.error("1. Created a 'data' directory with .txt files")
            st.error("2. Have sufficient permissions to create files")
            st.error("3. Have enough disk space")
            raise e

    def run(self):
        st.title("📚 Shakespeare Insights: AI-Powered Q&A")
        st.write("Explore the timeless works of Shakespeare with AI-powered insights! Ask questions about his plays, sonnets, and poems, and receive answers driven by the Complete Works of William Shakespeare.")

        # Sidebar to list all the works
        with st.sidebar:
            st.header("Shakespeare's Works")
            for work in works:
                st.write(work)

        # Question suggestions 
        st.subheader("Suggested Questions")
        cols = st.columns([4, 2, 2, 2, 2])   
        for idx, question in enumerate(suggested_questions):
            if cols[idx].button(question):
                st.session_state.selected_question = question

        # Question form 
        
        with st.form(key='qa_form'):
            query = st.text_area("Enter your question:", value=st.session_state.selected_question, height=100)
            submit_button = st.form_submit_button("Ask Question")

        if submit_button and query:
            self.process_query(query)

        self.display_history()

    def process_query(self, query: str):
        try:
            with st.spinner("Thinking... 🤔"):
                start_time = time.time()
                result = st.session_state.rag_engine.query(query)
                end_time = time.time()
                
                st.session_state.history.append({
                    "question": query,
                    "answer": result["answer"],
                    "sources": result["source_documents"],
                    "time": end_time - start_time
                })
                
        except Exception as e:
            st.error(f"Error processing query: {str(e)}")

    def display_history(self):
        if st.session_state.history:
            st.write("---")
            st.subheader("Conversation History")
            
            for i, item in enumerate(reversed(st.session_state.history)):
                with st.container():
                    st.write(f"**Q: {item['question']}**")
                    st.write(f"A: {item['answer']}")
                    
                    with st.expander("View Sources"):
                        for idx, doc in enumerate(item['sources']):
                            st.write(f"Source {idx + 1}:")
                            st.write(doc.page_content[:200] + "...")
                    
                    st.write(f"*Response time: {item['time']:.2f} seconds*")
                    st.write("---")

def main():
    st.set_page_config(
        page_title="Shakespeare Insights: AI-Powered Q&A",
        page_icon="📚",
        layout="wide"
    )
    
    interface = RAGWebInterface()
    interface.run()

if __name__ == "__main__":
    main()
