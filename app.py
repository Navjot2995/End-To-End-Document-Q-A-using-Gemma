import os
import time
from datetime import datetime
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
from typing import List, Tuple, Optional

# Constants
LOTTIE_URL = "https://lottie.host/f1216edb-4e09-46e5-8f1e-90c367b6fc13/iM4N0EXuvy.json"
TEMP_FOLDER = "./uploaded_pdfs"
DEFAULT_MODEL = "gemma2-9b-it"

# Initialize Streamlit page config
st.set_page_config(
    page_title="📘 Gemma Document Q&A",
    layout="wide",
    page_icon="📘",
    initial_sidebar_state="expanded"
)

@st.cache_data(show_spinner=False)
def load_lottie(url: str) -> Optional[dict]:
    """Load Lottie animation from URL"""
    try:
        res = requests.get(url, timeout=10)
        return res.json() if res.status_code == 200 else None
    except Exception:
        return None

def initialize_session_state():
    """Initialize all session state variables"""
    defaults = {
        "chat_history": [],
        "vectors": None,
        "embeddings": None,
        "user_groq_api_key": "",
        "processing_docs": False
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

def setup_environment():
    """Configure API keys and environment variables"""
    # Check Google API key (required)
    if 'GOOGLE_API_KEY' not in st.secrets:
        st.error("❌ Missing Google API key in secrets.toml")
        st.stop()
    os.environ['GOOGLE_API_KEY'] = st.secrets['GOOGLE_API_KEY']

    # Check Groq API key (user input or secrets)
    if st.session_state.user_groq_api_key:
        os.environ['GROQ_API_KEY'] = st.session_state.user_groq_api_key
    elif 'GROQ_API_KEY' in st.secrets:
        os.environ['GROQ_API_KEY'] = st.secrets['GROQ_API_KEY']
    else:
        st.error("🔑 Please provide a Groq API key in the sidebar")
        st.stop()

def apply_custom_styling():
    """Apply custom CSS styling"""
    st.markdown("""
    <style>
    /* Modern styling */
    .stTextInput>div>div>input, .stTextArea>div>textarea {
        border-radius: 12px !important;
        padding: 12px !important;
    }
    .stButton>button {
        border-radius: 12px !important;
        padding: 12px 24px !important;
        transition: all 0.3s ease !important;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    [data-testid="stSidebar"] {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
    }
    </style>
    """, unsafe_allow_html=True)

def process_uploaded_files(uploaded_files: List) -> bool:
    """Process PDF files and create vector store"""
    try:
        st.session_state.processing_docs = True
        st.session_state.embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        
        # Create temp directory if needed
        os.makedirs(TEMP_FOLDER, exist_ok=True)
        
        # Save and load PDFs
        docs = []
        progress_bar = st.progress(0)
        for i, uploaded_file in enumerate(uploaded_files):
            with open(os.path.join(TEMP_FOLDER, uploaded_file.name), "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            loader = PyPDFLoader(os.path.join(TEMP_FOLDER, uploaded_file.name))
            docs.extend(loader.load())
            progress_bar.progress((i + 1) / len(uploaded_files))
        
        if not docs:
            st.error("No text could be extracted from the PDF files")
            return False
        
        # Split and vectorize documents
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
        if 'progress_bar' in locals():
            progress_bar.empty()

def generate_response(user_question: str) -> Tuple[str, str]:
    """Generate AI response using RAG pipeline"""
    try:
        llm = ChatGroq(
            groq_api_key=os.environ['GROQ_API_KEY'],
            model_name=DEFAULT_MODEL,
            temperature=0.3
        )
        
        prompt = ChatPromptTemplate.from_template(
            """Answer the question based only on the provided context:
            
            Context:
            {context}
            
            Question: {input}
            
            Provide a concise, accurate response. If unsure, say "I couldn't find 
            that information in the documents"."""
        )
        
        document_chain = create_stuff_documents_chain(llm, prompt)
        retriever = st.session_state.vectors.as_retriever()
        retrieval_chain = create_retrieval_chain(retriever, document_chain)
        
        response = retrieval_chain.invoke({"input": user_question})
        return response['answer'], datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
    except Exception as e:
        st.error(f"AI response generation failed: {str(e)}")
        return None, None

def render_chat_history():
    """Display the chat conversation"""
    with st.container(height=500, border=True):
        for q, a, t in st.session_state.chat_history:
            st.chat_message("human").markdown(f"**You**: {q}")
            st.chat_message("ai").markdown(f"**Gemma**: {markdown2.markdown(a)}")
            st.caption(f"🕒 {t}")
            st.divider()

def main():
    # Initialize app
    initialize_session_state()
    animation = load_lottie(LOTTIE_URL)
    
    # Sidebar configuration
    with st.sidebar:
        st.title("Configuration")
        
        # API Key Section
        with st.expander("🔐 API Settings", expanded=True):
            st.session_state.user_groq_api_key = st.text_input(
                "Groq API Key",
                type="password",
                help="Get your key from https://console.groq.com/keys"
            )
        
        # Document Upload Section
        with st.expander("📄 Document Upload", expanded=True):
            uploaded_files = st.file_uploader(
                "Upload PDFs",
                type=["pdf"],
                accept_multiple_files=True,
                disabled=st.session_state.processing_docs
            )
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button("Process Documents", disabled=not uploaded_files):
                    if process_uploaded_files(uploaded_files):
                        st.success("Knowledge base created!")
            
            with col2:
                if st.button("Clear Cache", type="secondary"):
                    st.session_state.vectors = None
                    if os.path.exists(TEMP_FOLDER):
                        for f in os.listdir(TEMP_FOLDER):
                            os.remove(os.path.join(TEMP_FOLDER, f))
                    st.rerun()
    
    # Main content area
    st.title("📘 Document Intelligence Assistant")
    st.caption("Chat with your documents using Groq's lightning-fast AI")
    
    if animation:
        st_lottie(animation, speed=1, height=200)
    
    # Setup environment (checks API keys)
    setup_environment()
    
    # Chat interface
    render_chat_history()
    
    # User input
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
    apply_custom_styling()
    main()
