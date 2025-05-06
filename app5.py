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

# Initialize Streamlit page config
st.set_page_config(page_title="📘 Gemma Document Q&A", layout="wide", page_icon="📘")

@st.cache_data
def load_lottie_url(url):
    res = requests.get(url)
    if res.status_code == 200:
        return res.json()
    return None

# Load animation
animation = load_lottie_url("https://lottie.host/f1216edb-4e09-46e5-8f1e-90c367b6fc13/iM4N0EXuvy.json")

# --- SECURE API KEY HANDLING ---
def check_and_initialize_keys():
    """Verify required secrets are present and initialize the environment"""
    required_keys = ['GROQ_API_KEY', 'GOOGLE_API_KEY']
    missing_keys = [key for key in required_keys if key not in st.secrets]
    
    if missing_keys:
        st.error(f"Missing required API keys in secrets: {', '.join(missing_keys)}")
        st.stop()
    
    # Initialize the environment variables from secrets
    os.environ['GROQ_API_KEY'] = st.secrets['GROQ_API_KEY']
    os.environ['GOOGLE_API_KEY'] = st.secrets['GOOGLE_API_KEY']

# Check keys at startup
check_and_initialize_keys()

# --- UI THEMING ---
selected_theme = st.sidebar.radio("🎨 Select Theme", ["Dark", "Light"])
theme_css = """
.main { background-color: %s; color: %s; }
.stTextInput>div>div>input, .stTextArea>div>textarea {
    background-color: %s;
    color: %s;
    border: 1px solid #3b82f6;
    border-radius: 10px;
}
"""
theme_config = {
    "Dark": {
        "bg": "#0f172a",
        "text": "#f1f5f9",
        "input_bg": "#1e293b",
        "input_text": "white"
    },
    "Light": {
        "bg": "#f4f4f5",
        "text": "#111827",
        "input_bg": "#ffffff",
        "input_text": "#111827"
    }
}

current_theme = theme_config[selected_theme]
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600&display=swap');
html, body, [class*="css"] {{ font-family: 'Inter', sans-serif; }}
{theme_css % (
    current_theme['bg'],
    current_theme['text'],
    current_theme['input_bg'],
    current_theme['input_text']
)}
.stButton>button {{
    background-color: #3b82f6;
    color: white;
    border-radius: 10px;
    padding: 10px;
    transition: 0.3s ease-in-out;
}}
.stButton>button:hover {{
    background-color: #2563eb;
    transform: scale(1.05);
}}
.stMarkdown h1, h2, h3 {{ color: #38bdf8; }}
.uploadedFileName {{ color: #a5f3fc; }}
.chat-container {{
    max-height: 400px;
    overflow-y: auto;
    border: 1px solid #3b82f6;
    padding: 10px;
    border-radius: 10px;
    background-color: #1e293b;
    margin-bottom: 1rem;
}}
.chat-bubble {{
    border-radius: 12px;
    padding: 10px 15px;
    margin: 8px 0;
    width: fit-content;
    max-width: 85%;
}}
.user-bubble {{
    background-color: #3b82f6;
    color: white;
    align-self: flex-end;
    margin-left: auto;
}}
.bot-bubble {{
    background-color: #0f172a;
    color: white;
    align-self: flex-start;
}}
.timestamp {{
    font-size: 11px;
    color: #9ca3af;
    margin-top: 4px;
    margin-bottom: 6px;
    text-align: right;
}}
</style>
""", unsafe_allow_html=True)

if animation:
    st_lottie(animation, speed=1, height=220, key="header_anim")

# --- MAIN APP ---
st.title("🧠 Gemma PDF Analyzer")
st.markdown("Upload your PDFs, build a smart document database, and chat interactively with **Gemma**.")

# Initialize session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "vectors" not in st.session_state:
    st.session_state.vectors = None

# Sidebar controls
st.sidebar.header("📄 Upload PDFs")
uploaded_files = st.sidebar.file_uploader(
    "Choose your PDFs", 
    type=["pdf"], 
    accept_multiple_files=True,
    key="pdf_uploader"
)

if st.sidebar.button("🗑️ Clear Uploaded Files"):
    if os.path.exists("./uploaded_pdfs"):
        for f in os.listdir("./uploaded_pdfs"):
            os.remove(os.path.join("./uploaded_pdfs", f))
        st.sidebar.success("Uploaded files cleared!")
        if "vectors" in st.session_state:
            del st.session_state.vectors

def vector_embedding():
    if uploaded_files:
        try:
            st.session_state.embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
            temp_folder = "./uploaded_pdfs"
            os.makedirs(temp_folder, exist_ok=True)
            
            with st.spinner("Processing PDFs..."):
                progress = st.progress(0)
                for i, uploaded_file in enumerate(uploaded_files):
                    with open(os.path.join(temp_folder, uploaded_file.name), "wb") as f:
                        f.write(uploaded_file.getvalue())
                    progress.progress((i + 1) / len(uploaded_files))
                
                docs = []
                for f in os.listdir(temp_folder):
                    loader = PyPDFLoader(os.path.join(temp_folder, f))
                    docs.extend(loader.load())
                
                if not docs:
                    st.error("No text could be extracted from the PDF files.")
                    return
                
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1000,
                    chunk_overlap=200
                )
                final_documents = text_splitter.split_documents(docs)
                
                if not final_documents:
                    st.error("Document splitting failed. Try different files.")
                    return
                
                st.session_state.vectors = FAISS.from_documents(
                    final_documents, 
                    st.session_state.embeddings
                )
                st.toast("🎉 Knowledge base ready to use!", icon="📚")
                st.sidebar.success("✅ PDFs processed and vector DB ready!")
        except Exception as e:
            st.error(f"Error processing documents: {str(e)}")

if st.sidebar.button("📌 Build Knowledge Base"):
    vector_embedding()

# Chat interface
st.subheader("💬 Chat with your PDFs")

st.markdown("<div class='chat-container'>", unsafe_allow_html=True)
if st.session_state.chat_history:
    for q, a, t in reversed(st.session_state.chat_history):
        st.markdown(f"""
        <div class='chat-bubble user-bubble'>
            <strong>You:</strong><br>{q}<div class='timestamp'>{t}</div>
        </div>
        <div class='chat-bubble bot-bubble'>
            <strong>Gemma:</strong><br>{markdown2.markdown(a)}<div class='timestamp'>{t}</div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("<script>window.scrollTo(0, document.body.scrollHeight);</script>", unsafe_allow_html=True)
        
st.markdown("</div>", unsafe_allow_html=True)

user_question = st.text_input("Ask another question:", key="chat_input")

if st.button("💡 Send") and user_question:
    if st.session_state.vectors is None:
        st.warning("Please build the knowledge base first!")
    else:
        with st.spinner("Gemma is generating a response..."):
            try:
                llm = ChatGroq(
                    groq_api_key=os.environ['GROQ_API_KEY'],
                    model_name="gemma2-9b-it",
                    temperature=0.3
                )
                
                prompt = ChatPromptTemplate.from_template(
                    """Answer the questions based on the provided context only.
                    Please provide the most accurate response based on the question.
                    <context>
                    {context}
                    </context>
                    Question: {input}"""
                )
                
                start = time.time()
                document_chain = create_stuff_documents_chain(llm, prompt)
                retriever = st.session_state.vectors.as_retriever()
                retrieval_chain = create_retrieval_chain(retriever, document_chain)
                response = retrieval_chain.invoke({"input": user_question})
                end = time.time()
                
                st.session_state.chat_history.append((
                    user_question,
                    response['answer'],
                    datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                ))
                st.experimental_rerun()
            except Exception as e:
                st.error(f"Error generating response: {str(e)}")

st.markdown("""
---
<p style='text-align: center; font-size: 14px;'>Built with 💙 using LangChain + Gemini + Streamlit</p>
""", unsafe_allow_html=True)