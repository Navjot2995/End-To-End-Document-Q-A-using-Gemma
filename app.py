import os
import time
import base64
from datetime import datetime
from typing import List, Tuple, Optional
import streamlit as st
from langchain_groq import ChatGroq
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from streamlit_lottie import st_lottie
import requests
import markdown2
from dotenv import load_dotenv

# Constants
LOTTIE_URL = "https://lottie.host/f1216edb-4e09-46e5-8f1e-90c367b6fc13/iM4N0EXuvy.json"
TEMP_FOLDER = "./uploaded_pdfs"
DEFAULT_MODEL = "gemma2-9b-it"

# --- Helper Functions ---
@st.cache_data(show_spinner=False)
def load_lottie(url: str) -> Optional[dict]:
    """Load Lottie animation from URL"""
    try:
        res = requests.get(url, timeout=10)
        return res.json() if res.status_code == 200 else None
    except Exception:
        return None

def get_secret(key: str) -> str:
    """Get secret from environment variables or Streamlit secrets"""
    return os.getenv(key) or st.secrets.get(key, "")

def get_base64_image(image_path: str) -> str:
    """Convert image to base64 for CSS embedding"""
    with open(image_path, "rb") as image_file:
        return f"data:image/png;base64,{base64.b64encode(image_file.read()).decode()}"

# --- Core Functions ---
def initialize_session_state():
    """Initialize all session state variables"""
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "vectors" not in st.session_state:
        st.session_state.vectors = None
    if "embeddings" not in st.session_state:
        st.session_state.embeddings = None
    if "processing_docs" not in st.session_state:
        st.session_state.processing_docs = False
    if "theme" not in st.session_state:
        st.session_state.theme = "dark"

def validate_api_keys():
    """Validate required API keys"""
    required_keys = {
        "GOOGLE_API_KEY": "Google API Key",
        "GROQ_API_KEY": "Groq API Key"
    }
    
    missing_keys = []
    for key, name in required_keys.items():
        if not get_secret(key):
            missing_keys.append(name)
    
    if missing_keys:
        st.error(f"Missing required API keys: {', '.join(missing_keys)}")
        st.stop()
    
    os.environ["GOOGLE_API_KEY"] = get_secret("GOOGLE_API_KEY")
    os.environ["GROQ_API_KEY"] = get_secret("GROQ_API_KEY")

def process_uploaded_files(uploaded_files: List) -> bool:
    """Process PDF files and create vector store"""
    try:
        st.session_state.processing_docs = True
        st.session_state.embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        
        os.makedirs(TEMP_FOLDER, exist_ok=True)
        docs = []
        
        with st.spinner("Processing documents..."):
            progress_bar = st.progress(0)
            for i, uploaded_file in enumerate(uploaded_files):
                file_path = os.path.join(TEMP_FOLDER, uploaded_file.name)
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                
                loader = PyPDFLoader(file_path)
                docs.extend(loader.load())
                progress_bar.progress((i + 1) / len(uploaded_files))
            
            if not docs:
                st.error("No text could be extracted from the PDF files")
                return False
            
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200
            )
            final_documents = text_splitter.split_documents(docs)
            
            st.session_state.vectors = FAISS.from_documents(
                final_documents, 
                st.session_state.embeddings
            )
            return True
            
    except Exception as e:
        st.error(f"Document processing failed: {str(e)}")
        return False
    finally:
        st.session_state.processing_docs = False

def generate_response(user_question: str) -> Tuple[Optional[str], Optional[str]]:
    """Generate AI response using RAG pipeline"""
    try:
        llm = ChatGroq(
            groq_api_key=os.environ["GROQ_API_KEY"],
            model_name=DEFAULT_MODEL,
            temperature=0.3
        )
        
        prompt = ChatPromptTemplate.from_template(
            """Answer the question based only on the provided context:
            Context: {context}
            Question: {input}
            Provide a concise, accurate response with proper markdown formatting."""
        )
        
        document_chain = create_stuff_documents_chain(llm, prompt)
        retriever = st.session_state.vectors.as_retriever()
        retrieval_chain = create_retrieval_chain(retriever, document_chain)
        
        response = retrieval_chain.invoke({"input": user_question})
        return response["answer"], datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
    except Exception as e:
        st.error(f"AI response generation failed: {str(e)}")
        return None, None

def render_chat_history():
    """Display the chat conversation"""
    chat_container = st.container(height=500)
    with chat_container:
        for q, a, t in st.session_state.chat_history:
            with st.chat_message("user"):
                st.markdown(q)
                st.caption(t)
            
            with st.chat_message("assistant"):
                st.markdown(markdown2.markdown(a), unsafe_allow_html=True)
                st.caption(t)
            
            st.divider()

# --- Main App ---
def main():
    # Initialize app
    st.set_page_config(
        page_title="Document Intelligence Assistant",
        layout="wide",
        page_icon="📘",
        initial_sidebar_state="expanded"
    )
    
    # Apply custom CSS
    st.markdown("""
    <style>
        .stApp { background-color: #0f172a; }
        [data-testid="stSidebar"] > div:first-child { background-color: #0f172a !important; }
        .stSidebar .sidebar-content { background-color: rgba(15, 23, 42, 0.9) !important; }
        .stFileUploader { background-color: #1e293b !important; border-radius: 10px !important; }
        .stButton>button { background-color: #3b82f6 !important; border: none !important; }
    </style>
    """, unsafe_allow_html=True)
    
    initialize_session_state()
    validate_api_keys()
    
    # Load animation
    animation = load_lottie(LOTTIE_URL)
    
    # Sidebar
    with st.sidebar:
        st.title("Document Intelligence Assistant")
        st.markdown("Chat with your documents using Groq's lightning-fast AI")
        
        with st.expander("⚙️ Settings", expanded=True):
            st.session_state.theme = st.radio(
                "Theme",
                ["dark", "light"],
                index=0 if st.session_state.theme == "dark" else 1,
                key="theme_selector"
            )
            
            uploaded_files = st.file_uploader(
                "Upload PDFs",
                type=["pdf"],
                accept_multiple_files=True,
                disabled=st.session_state.processing_docs,
                key="pdf_uploader"
            )
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button(
                    "Process Documents",
                    disabled=not uploaded_files or st.session_state.processing_docs,
                    key="process_button"
                ) and uploaded_files:
                    if process_uploaded_files(uploaded_files):
                        st.success("Knowledge base created!")
            
            with col2:
                if st.button(
                    "Clear Cache",
                    type="secondary",
                    disabled=st.session_state.processing_docs,
                    key="clear_button"
                ):
                    st.session_state.vectors = None
                    if os.path.exists(TEMP_FOLDER):
                        for f in os.listdir(TEMP_FOLDER):
                            os.remove(os.path.join(TEMP_FOLDER, f))
                    st.rerun()
    
    # Main content
    st.title("📘 Document Intelligence Assistant")
    
    if animation:
        st_lottie(animation, speed=1, height=200)
    
    render_chat_history()
    
    if prompt := st.chat_input("Ask about your documents..."):
        if not st.session_state.vectors:
            st.warning("Please upload and process documents first")
        else:
            with st.spinner("Generating response..."):
                answer, timestamp = generate_response(prompt)
                if answer:
                    st.session_state.chat_history.append((prompt, answer, timestamp))
                    st.rerun()

if __name__ == "__main__":
    load_dotenv()
    main()
