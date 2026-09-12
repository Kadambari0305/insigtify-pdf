import streamlit as st
import google.generativeai as genai
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from pypdf import PdfReader
from PIL import Image
from typing import List, Optional
import logging
import os
import io
import uuid
from datetime import datetime
from dotenv import load_dotenv

# Import Database Manager
from db_manager import db_manager

load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# Set page config
st.set_page_config(
    page_title="Multi-Document & Photo Analysis Hub",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
    <style>
    .main-title {
        text-align: center;
        padding: 0.8rem 0;
        color: #1E3D59;
        font-size: 2.3rem;
        font-weight: 800;
        background: linear-gradient(120deg, #1E3D59, #17B794, #FF6F61);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .badge {
        display: inline-block;
        padding: 0.25rem 0.6rem;
        font-size: 0.85rem;
        font-weight: 600;
        border-radius: 6px;
        margin-right: 0.4rem;
        margin-bottom: 0.4rem;
        color: white;
    }
    .badge-pdf { background-color: #e53935; }
    .badge-image { background-color: #1e88e5; }
    .auth-box {
        max-width: 480px;
        margin: 2rem auto;
        padding: 2rem;
        border-radius: 12px;
        background: #ffffff;
        box-shadow: 0 8px 24px rgba(0,0,0,0.1);
        border: 1px solid #e0e0e0;
    }
    .card {
        padding: 1.5rem;
        border-radius: 12px;
        background: #ffffff;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.08);
        margin-bottom: 1rem;
        border: 1px solid #e0e0e0;
    }
    .history-card {
        padding: 0.8rem;
        border-radius: 8px;
        background: #f1f5f9;
        margin-bottom: 0.5rem;
        border-left: 4px solid #17B794;
    }
    .footer {
        text-align: center;
        padding: 1.2rem;
        color: #666;
        font-size: 0.9rem;
        background: #f8f9fa;
        border-radius: 12px;
        margin-top: 2rem;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize Session States
if 'current_user' not in st.session_state:
    st.session_state.current_user = None
if 'api_key' not in st.session_state:
    st.session_state.api_key = os.getenv("GOOGLE_API_KEY", "")
if 'active_session_id' not in st.session_state:
    st.session_state.active_session_id = str(uuid.uuid4())
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'docs_processed' not in st.session_state:
    st.session_state.docs_processed = False
if 'vector_store' not in st.session_state:
    st.session_state.vector_store = None
if 'processed_files_summary' not in st.session_state:
    st.session_state.processed_files_summary = []
if 'image_previews' not in st.session_state:
    st.session_state.image_previews = {}

# Prompt Template
system_prompt = """
You are an expert multi-modal document analyst and research assistant.
You are analyzing information extracted from multiple sources including printed PDFs, scanned handwritten documents, receipts, invoices, and photo images.

Guidelines for your response:
1. Answer the user's question clearly, thoroughly, and accurately based on the provided document context.
2. ALWAYS cite the source file name(s) (e.g. `[Source: document.pdf]`) when presenting facts, data, or answers.
3. If information comes from multiple files, highlight the connections or compare data across files.
4. If a question cannot be answered from the provided context, state that clearly without making up facts.
"""

def extract_text_from_pdf(uploaded_file, api_key: str) -> str:
    """Extract text from standard PDF, fallback to Gemini OCR for scanned pages."""
    pdf_reader = PdfReader(uploaded_file)
    extracted_text = ""
    
    for page_num, page in enumerate(pdf_reader.pages, 1):
        text = page.extract_text()
        if text:
            extracted_text += f"\n--- Page {page_num} ---\n" + text

    if len(extracted_text.strip()) < 50:
        logger.info(f"PDF {uploaded_file.name} appears scanned. Using Gemini Vision OCR...")
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-1.5-flash')
        pdf_bytes = uploaded_file.getvalue()
        pdf_parts = [{"mime_type": "application/pdf", "data": pdf_bytes}]
        ocr_prompt = "Perform full OCR extraction on this scanned PDF document. Extract all printed text, handwritten notes, tables, and values in full detail."
        response = model.generate_content([ocr_prompt, pdf_parts[0]])
        extracted_text = response.text

    return extracted_text

def extract_text_from_image(uploaded_file, api_key: str) -> tuple[str, Image.Image]:
    """Perform Gemini Vision OCR on photos/images."""
    image = Image.open(uploaded_file)
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('gemini-1.5-flash')
    
    ocr_prompt = (
        "Analyze this photo/image in full detail. Extract all readable text, printed words, "
        "handwritten notes, figures, table contents, key-value pairs, and describe key visual data."
    )
    response = model.generate_content([ocr_prompt, image])
    return response.text, image

# --- AUTHENTICATION SCREEN ---
if not st.session_state.current_user:
    st.markdown("<h1 class='main-title'>⚡ Data Analysis Hub</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align:center; color:#555;'>Please Login or Create an Account to access your Multi-Modal Document Hub</p>", unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        tab_login, tab_signup = st.tabs(["🔐 Login", "📝 Sign Up"])

        with tab_login:
            st.markdown("### Welcome Back!")
            login_username = st.text_input("Username", key="login_user")
            login_password = st.text_input("Password", type="password", key="login_pwd")

            if st.button("Log In", type="primary", use_container_width=True):
                if login_username and login_password:
                    ok, msg = db_manager.authenticate_user(login_username, login_password)
                    if ok:
                        st.session_state.current_user = login_username.lower().strip()
                        st.session_state.active_session_id = str(uuid.uuid4())
                        st.session_state.chat_history = []
                        st.success(f"Welcome back, {login_username}!")
                        st.rerun()
                    else:
                        st.error(msg)
                else:
                    st.warning("Please enter username and password.")

        with tab_signup:
            st.markdown("### Create New Client Account")
            signup_username = st.text_input("Username", key="signup_user")
            signup_email = st.text_input("Email", key="signup_email")
            signup_password = st.text_input("Password", type="password", key="signup_pwd")

            if st.button("Create Account", type="primary", use_container_width=True):
                if signup_username and signup_password:
                    ok, msg = db_manager.register_user(signup_username, signup_email, signup_password)
                    if ok:
                        st.success("Account created successfully! You can now log in.")
                    else:
                        st.error(msg)
                else:
                    st.warning("Please fill in all required fields.")

    st.stop()

# --- AUTHENTICATED APP SCREEN ---
with st.sidebar:
    st.markdown(f"👤 **Client:** `{st.session_state.current_user}`")
    if st.button("🚪 Logout", use_container_width=True):
        st.session_state.current_user = None
        st.session_state.docs_processed = False
        st.session_state.vector_store = None
        st.session_state.chat_history = []
        st.rerun()

    st.markdown("---")
    st.header("🔑 API & Configuration")
    default_key = os.getenv("GOOGLE_API_KEY", "")
    st.session_state.api_key = st.text_input("Google Gemini API Key", value=st.session_state.api_key or default_key, type="password")

    st.markdown("---")
    st.header("📜 Chat Session History")
    
    # Reload Past Conversations
    user_sessions = db_manager.get_user_sessions(st.session_state.current_user)
    if user_sessions:
        for sess in user_sessions:
            sess_id = sess["session_id"]
            title = sess.get("title", "Chat Session")
            msg_count = len(sess.get("messages", []))
            
            is_active = (sess_id == st.session_state.active_session_id)
            btn_label = f"{"👉 " if is_active else "💬 "}{title[:24]} ({msg_count} msgs)"
            
            if st.button(btn_label, key=f"sess_{sess_id}", use_container_width=True):
                st.session_state.active_session_id = sess_id
                st.session_state.chat_history = sess.get("messages", [])
                st.session_state.processed_files_summary = [
                    {"name": fname, "type": "doc", "chunks": "-"} for fname in sess.get("files", [])
                ]
                st.rerun()
    else:
        st.caption("No past saved sessions found.")

    if st.button("➕ Start New Session", type="secondary", use_container_width=True):
        st.session_state.active_session_id = str(uuid.uuid4())
        st.session_state.chat_history = []
        st.session_state.docs_processed = False
        st.session_state.vector_store = None
        st.session_state.processed_files_summary = []
        st.rerun()

# Title Header
st.markdown("<h1 class='main-title'>⚡ Data Analysis Hub: PDFs, OCR & Photos</h1>", unsafe_allow_html=True)

# Layout Tabs
tab_upload, tab_inspect, tab_chat = st.tabs(["📁 File Uploader", "🖼️ Photo & PDF Inspector", "💬 AI Assistant & History"])

with tab_upload:
    st.markdown("""
        <div class='card'>
            <h3>📤 Upload Documents & Photos</h3>
            <p>Upload multiple PDFs, scanned documents, receipts, handwritten notes, or images to analyze together.</p>
        </div>
    """, unsafe_allow_html=True)

    uploaded_files = st.file_uploader(
        "Choose PDF files or Images",
        type=['pdf', 'png', 'jpg', 'jpeg', 'webp'],
        accept_multiple_files=True,
        key='multi_uploader'
    )

    if uploaded_files:
        st.write(f"**Selected Files ({len(uploaded_files)}):**")
        cols = st.columns(min(len(uploaded_files), 4))
        for idx, file in enumerate(uploaded_files):
            col = cols[idx % 4]
            is_pdf = file.name.lower().endswith('.pdf')
            badge_class = "badge-pdf" if is_pdf else "badge-image"
            file_type = "PDF" if is_pdf else "IMAGE"
            col.markdown(f"<span class='badge {badge_class}'>{file_type}</span> <b>{file.name}</b>", unsafe_allow_html=True)

        if st.button("🚀 Process All Files", type="primary"):
            if not st.session_state.api_key:
                st.warning("⚠️ Please enter your Google Gemini API Key in the sidebar.")
            else:
                with st.spinner("Processing files with AI OCR & Vector Indexing..."):
                    try:
                        all_langchain_docs = []
                        processed_summary = []
                        image_previews = {}

                        splitter = RecursiveCharacterTextSplitter(
                            chunk_size=CHUNK_SIZE,
                            chunk_overlap=CHUNK_OVERLAP,
                            length_function=len
                        )

                        for file in uploaded_files:
                            file_name = file.name
                            ext = file_name.split('.')[-1].lower()

                            st.write(f"⚙️ Processing `{file_name}`...")

                            if ext == 'pdf':
                                raw_text = extract_text_from_pdf(file, st.session_state.api_key)
                                file_kind = 'pdf'
                            else:
                                raw_text, img = extract_text_from_image(file, st.session_state.api_key)
                                image_previews[file_name] = img
                                file_kind = 'image'

                            text_chunks = splitter.split_text(raw_text)

                            for chunk_idx, chunk in enumerate(text_chunks):
                                all_langchain_docs.append(Document(
                                    page_content=chunk,
                                    metadata={"source": file_name, "type": file_kind, "chunk": chunk_idx + 1}
                                ))

                            processed_summary.append({
                                "name": file_name,
                                "type": file_kind,
                                "chunks": len(text_chunks)
                            })

                        # Unified FAISS Vector Store
                        embeddings = GoogleGenerativeAIEmbeddings(
                            model="models/text-embedding-004",
                            google_api_key=st.session_state.api_key
                        )
                        vector_store = FAISS.from_documents(all_langchain_docs, embedding=embeddings)
                        vector_store.save_local("faiss-index")

                        st.session_state.vector_store = vector_store
                        st.session_state.docs_processed = True
                        st.session_state.processed_files_summary = processed_summary
                        st.session_state.image_previews = image_previews

                        st.success(f"✅ Successfully processed {len(uploaded_files)} file(s) into {len(all_langchain_docs)} vector chunks!")
                    except Exception as e:
                        logger.error(f"Processing error: {e}", exc_info=True)
                        err_str = str(e)
                        if "401" in err_str or "ACCESS_TOKEN_TYPE_UNSUPPORTED" in err_str or "invalid authentication" in err_str.lower():
                            st.error(
                                "🔑 **Authentication Error (401 - Invalid API Key)**\n\n"
                                "Your `GOOGLE_API_KEY` is invalid or using an unsupported token format.\n"
                                "- Please get a standard Google Gemini API Key starting with `AIzaSy...` from [Google AI Studio](https://aistudio.google.com/app/apikey).\n"
                                "- Paste your valid key into the sidebar under **🔑 API & Configuration** or set the `GOOGLE_API_KEY` environment variable in your server deployment."
                            )
                        else:
                            st.error(f"Error processing files: {err_str}")

with tab_inspect:
    st.header("🖼️ Uploaded Photo & Image Inspector")
    if st.session_state.image_previews:
        cols = st.columns(3)
        for idx, (img_name, img_obj) in enumerate(st.session_state.image_previews.items()):
            col = cols[idx % 3]
            col.image(img_obj, caption=img_name, use_container_width=True)
    else:
        st.info("No photo previews available. Upload images in the File Uploader tab to inspect them here.")

with tab_chat:
    st.header("💬 AI Assistant & Session Chat")

    # Render Existing History
    if st.session_state.chat_history:
        st.markdown("### 📜 Session Conversation Thread")
        for msg in st.session_state.chat_history:
            role = msg.get("role")
            content = msg.get("content")
            if role == "user":
                st.chat_message("user").write(content)
            else:
                st.chat_message("assistant").write(content)
        st.markdown("---")

    if st.session_state.docs_processed:
        st.info("💡 You can ask questions about any individual file or cross-compare information across all uploaded files and photos.")
        user_question = st.text_area("🤔 Ask a question across your documents & photos:", height=100)

        if st.button("Send Question", type="primary"):
            if user_question.strip():
                with st.spinner("Analyzing context with Gemini..."):
                    try:
                        # Load vector store if needed
                        if st.session_state.vector_store is None:
                            embeddings = GoogleGenerativeAIEmbeddings(
                                model="models/text-embedding-004",
                                google_api_key=st.session_state.api_key
                            )
                            if os.path.exists("faiss-index"):
                                st.session_state.vector_store = FAISS.load_local(
                                    "faiss-index",
                                    embeddings,
                                    allow_dangerous_deserialization=True
                                )
                            else:
                                st.error("Vector store index not found. Please process documents first.")
                                st.stop()

                        # Similarity Search
                        relevant_docs = st.session_state.vector_store.similarity_search(user_question, k=6)

                        context_parts = []
                        cited_sources = set()
                        for doc in relevant_docs:
                            src = doc.metadata.get("source", "Unknown")
                            cited_sources.add(src)
                            context_parts.append(f"--- [Source: {src}] ---\n{doc.page_content}")

                        combined_context = "\n\n".join(context_parts)

                        # Generate Answer
                        genai.configure(api_key=st.session_state.api_key)
                        model = genai.GenerativeModel('gemini-1.5-flash')

                        prompt = f"""
                        {system_prompt}

                        Extracted Context from Uploaded Files & Photos:
                        {combined_context}

                        User Question: {user_question}
                        """

                        response = model.generate_content(prompt)
                        answer_text = response.text

                        # Display Assistant Answer
                        st.chat_message("user").write(user_question)
                        st.chat_message("assistant").write(answer_text)

                        # Save Turn to Database Manager
                        file_names = [f["name"] for f in st.session_state.processed_files_summary]
                        session_title = user_question[:30] + "..." if len(user_question) > 30 else user_question
                        
                        db_manager.save_chat_turn(
                            username=st.session_state.current_user,
                            session_id=st.session_state.active_session_id,
                            session_title=session_title,
                            files=file_names,
                            user_msg=user_question,
                            assistant_msg=answer_text
                        )

                        # Update Local State
                        st.session_state.chat_history.append({"role": "user", "content": user_question, "timestamp": datetime.now().isoformat()})
                        st.session_state.chat_history.append({"role": "assistant", "content": answer_text, "timestamp": datetime.now().isoformat()})
                        st.rerun()

                    except Exception as e:
                        err_str = str(e)
                        if "401" in err_str or "ACCESS_TOKEN_TYPE_UNSUPPORTED" in err_str or "invalid authentication" in err_str.lower():
                            st.error(
                                "🔑 **Authentication Error (401 - Invalid API Key)**\n\n"
                                "Your `GOOGLE_API_KEY` is invalid or using an unsupported token format.\n"
                                "- Please get a standard Google Gemini API Key starting with `AIzaSy...` from [Google AI Studio](https://aistudio.google.com/app/apikey).\n"
                                "- Paste your valid key into the sidebar under **🔑 API & Configuration** or set the `GOOGLE_API_KEY` environment variable in your server deployment."
                            )
                        else:
                            st.error(f"Error generating answer: {err_str}")
            else:
                st.warning("Please type a question before sending.")
    else:
        st.warning("👈 Please upload and click 'Process All Files' in the File Uploader tab first.")

# Footer
st.markdown("""
    <div class="footer">
        <p>Built with ❤ for Multi-Modal Document & Photo Analysis | Powered by Gemini 3.6 & MongoDB</p>
    </div>
""", unsafe_allow_html=True)