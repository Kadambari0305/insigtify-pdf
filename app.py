import streamlit as st
import google.generativeai as genai
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from pypdf import PdfReader
from typing import List, Optional
import logging
import os

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Define constants
CHUNK_SIZE = 1000  # Example chunk size
CHUNK_OVERLAP = 200  # Example overlap
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB max file size

# Set page config
st.set_page_config(
    page_title="Data Analysis Hub",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Onboarding / Welcome Card ---

st.markdown("""
    <div style="background-color: #e0f7fa; padding: 1.5rem; border-radius: 10px; margin-bottom: 2rem;">
        <h2 style="color: #00796b;">👋 Welcome to Data Analysis Hub!</h2>
        <p style="color: #004d40; font-size: 1.1rem;">
            Easily upload your business documents, scanned notes, or PDFs.<br>
            Our AI will intelligently analyze and answer your questions! 🚀
        </p>
        <ul style="color: #00695c; font-size: 1rem;">
            <li>Step 1️⃣: Enter your Google Gemini API Key in sidebar.</li>
            <li>Step 2️⃣: Upload one or more PDF files.</li>
            <li>Step 3️⃣: Click "Analyze" and ask questions!</li>
        </ul>
    </div>
""", unsafe_allow_html=True)

# Custom styles
st.markdown("""
    <style>
    /* Main title */
    .main-title {
        text-align: center;
        padding: 1.5rem 0;
        color: #1E3D59;
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(120deg, #1E3D59, #17B794);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    /* Cards */
    .card {
        padding: 2rem;
        border-radius: 15px;
        background: #ffffff;
        box-shadow: 0 8px 16px rgba(0, 0, 0, 0.1);
        margin: 1.5rem 0;
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    .card:hover {
        transform: translateY(-10px);
        box-shadow: 0 12px 24px rgba(0, 0, 0, 0.2);
    }
    .card-title {
        color: #1E3D59;
        font-size: 1.75rem;
        margin-bottom: 1.25rem;
        font-weight: 600;
    }
    .card-text {
        color: #666;
        font-size: 1.1rem;
        margin-bottom: 1.75rem;
    }

    /* Buttons */
    .stButton button {
        background: linear-gradient(135deg, #17B794, #1E3D59);
        color: white;
        border: none;
        padding: 0.75rem 1.5rem;
        border-radius: 8px;
        font-size: 1rem;
        font-weight: 600;
        cursor: pointer;
        transition: background 0.3s ease;
    }
    .stButton button:hover {
        background: linear-gradient(135deg, #1E3D59, #17B794);
    }

    /* Sidebar */
    .sidebar .sidebar-content {
        background: linear-gradient(145deg, #1E3D59, #17B794);
        color: white;
        padding: 1.5rem;
        border-radius: 15px;
    }
    .sidebar .sidebar-content h1 {
        color: white;
        font-size: 1.75rem;
        margin-bottom: 1.5rem;
    }
    .sidebar .sidebar-content h2 {
        color: white;
        font-size: 1.5rem;
        margin-bottom: 1rem;
    }
    .sidebar .sidebar-content p {
        color: #f0f0f0;
        font-size: 1rem;
        margin-bottom: 1rem;
    }

    /* File uploader */
    .stFileUploader {
        margin-bottom: 1.5rem;
    }

    /* Text input */
    .stTextInput input {
        border: 2px solid #1E3D59;
        border-radius: 8px;
        padding: 0.75rem;
        font-size: 1rem;
    }

    /* Spinner */
    .stSpinner {
        color: #17B794;
    }

    /* Footer */
    .footer {
        text-align: center;
        padding: 1.5rem;
        color: #666;
        font-size: 0.9rem;
        background: #f0f0f0;
        border-radius: 15px;
        margin-top: 2rem;
    }
    </style>
    """, unsafe_allow_html=True)

# Initialize session state
if 'api_key' not in st.session_state:
    st.session_state.api_key = None
if 'docs_processed' not in st.session_state:
    st.session_state.docs_processed = False
if 'vector_store' not in st.session_state:
    st.session_state.vector_store = None
if 'current_pdf' not in st.session_state:
    st.session_state.current_pdf = None
if 'current_mode' not in st.session_state:
    st.session_state.current_mode = None  # Tracks whether normal or OCR PDF is being processed

# Sidebar for API key and instructions
with st.sidebar:
    st.header("API Configuration")
    st.session_state.api_key = st.text_input("Google Gemini API Key", type="password")
    st.markdown("---")
    st.markdown("### 📋 Instructions")
    st.markdown("""
    1. Upload one or more PDF files
    2. Click 'Process Documents'
    3. Ask questions about your documents
    """)

# Main title
st.markdown("<h1 class='main-title'>Data Analysis Hub</h1>", unsafe_allow_html=True)

# Create two columns for the cards
col1, col2 = st.columns(2)

# Expert system prompt
input_prompt = """
You are an expert in document analysis and information extraction, specializing in:
- Printed business documents (invoices, receipts, purchase orders)
- Handwritten notes and records
- Scanned historical documents
- Mixed format documents containing both printed and handwritten elements

Your capabilities include:
- Reading and interpreting handwritten text, including different handwriting styles and varying legibility
- Processing traditional printed/typed documents
- Understanding document structure regardless of formatting or layout
- Handling various document qualities (faded text, smudges, creases, watermarks)
- Working with different languages and numerical formats
- Recognizing common business terms, financial data, and document-specific terminology

For any document you analyze:
1. Consider both printed and handwritten elements equally
2. Account for potential quality issues in scanned documents
3. Look for contextual clues to validate information
4. Flag any uncertainties or illegible portions
5. Maintain accuracy while dealing with different writing styles and formats

Please answer questions based on the information visible in the provided document.
"""

def get_gemini_response(input_prompt, pdf_data, user_question):
    """Get response from Gemini model"""
    model = genai.GenerativeModel('gemini-2.0-flash')

    # Combine the prompts
    combined_prompt = f"""
    {input_prompt}

    User Question: {user_question}
    """

    response = model.generate_content([combined_prompt, pdf_data[0]])
    return response.text

def process_pdf(uploaded_file):
    """Process the uploaded PDF file for the model"""
    if uploaded_file is not None:
        # Read the file into bytes
        bytes_data = uploaded_file.getvalue()

        pdf_parts = [
            {
                "mime_type": uploaded_file.type,
                "data": bytes_data
            }
        ]
        return pdf_parts
    else:
        raise FileNotFoundError("No file uploaded")

with col1:
    # Normal PDF Analysis UI
    st.markdown("""
        <div class='card' style="color: black;">
            <h3 class='card-title'>📄 Normal PDF Analysis</h3>
            <p class='card-text'>Upload PDF files for standard text extraction and analysis.
            </p>
        </div>
    """, unsafe_allow_html=True)
    normal_pdf_file = st.file_uploader("Choose a PDF file", type=['pdf'], key='normal_pdf')
    if st.button("Analyze Normal PDF"):
        if normal_pdf_file is not None:
            if st.session_state.api_key:
                # Check if a new PDF is uploaded or mode is switched
                if (st.session_state.current_pdf != normal_pdf_file.name or
                    st.session_state.current_mode != "normal"):
                    st.session_state.current_pdf = normal_pdf_file.name
                    st.session_state.current_mode = "normal"
                    st.session_state.docs_processed = False
                    st.session_state.vector_store = None

                with st.spinner("Processing documents..."):
                    try:
                        # Extract text from PDF
                        pdf_reader = PdfReader(normal_pdf_file)
                        raw_text = ""
                        for page in pdf_reader.pages:
                            raw_text += page.extract_text()

                        # Create text chunks
                        splitter = RecursiveCharacterTextSplitter(
                            chunk_size=CHUNK_SIZE,
                            chunk_overlap=CHUNK_OVERLAP,
                            length_function=len
                        )
                        text_chunks = splitter.split_text(raw_text)

                        # Create and save vector store
                        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=st.session_state.api_key)
                        vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
                        vector_store.save_local("faiss-index")

                        st.session_state.vector_store = vector_store
                        st.session_state.docs_processed = True
                        st.success("✅ Documents processed successfully!")
                    except Exception as e:
                        st.error(f"Error processing documents: {str(e)}")
            else:
                st.warning("Please enter your Google Gemini API Key.")
        else:
            st.warning("Please upload a PDF file first.")

with col2:
    # OCR PDF Analysis UI
    st.markdown("""
        <div class='card' style="color: black;">
            <h3 class='card-title'>🖹 OCR PDF Analysis</h3>
            <p class='card-text'>Upload scanned PDFs or images within PDFs for Optical Character Recognition (OCR) based text extraction.</p>
        </div>
    """, unsafe_allow_html=True)
    ocr_pdf_file = st.file_uploader("Choose a PDF file for OCR", type=['pdf'], key='ocr_pdf')
    if st.button("Analyze OCR PDF"):
        if ocr_pdf_file is not None:
            if st.session_state.api_key:
                # Check if a new PDF is uploaded or mode is switched
                if (st.session_state.current_pdf != ocr_pdf_file.name or
                    st.session_state.current_mode != "ocr"):
                    st.session_state.current_pdf = ocr_pdf_file.name
                    st.session_state.current_mode = "ocr"
                    st.session_state.docs_processed = False
                    st.session_state.vector_store = None

                with st.spinner("Processing documents..."):
                    try:
                        # Process the PDF and send it directly to the LLM
                        pdf_data = process_pdf(ocr_pdf_file)
                        st.session_state.ocr_response = pdf_data
                        st.session_state.docs_processed = True
                        st.success("✅ Documents processed successfully!")
                    except Exception as e:
                        st.error(f"Error processing documents: {str(e)}")
            else:
                st.warning("Please enter your Google Gemini API Key.")
        else:
            st.warning("Please upload a PDF file first.")

# Chat interface
if st.session_state.docs_processed:
    user_question = st.text_area("🤔 Ask a question about your documents:", height=100)
    if st.button("Send", type="primary"):
        if user_question:
            with st.spinner("Thinking..."):
                try:
                    if st.session_state.current_mode == "normal":
                        # Load vector store if it exists
                        if st.session_state.vector_store is None:
                            embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=st.session_state.api_key)
                            if os.path.exists("faiss-index"):
                                st.session_state.vector_store = FAISS.load_local("faiss-index", embeddings, allow_dangerous_deserialization=True)
                            else:
                                st.error("Vector store not found. Please process the documents first.")
                                st.stop()

                        # Perform similarity search to find relevant text chunks
                        relevant_chunks = st.session_state.vector_store.similarity_search(user_question, k=3)  # Get top 3 relevant chunks
                        context = "\n".join([chunk.page_content for chunk in relevant_chunks])

                        # Configure Gemini and generate a natural language response
                        genai.configure(api_key=st.session_state.api_key)
                        model = genai.GenerativeModel('gemini-2.0-flash')
                        prompt = f"""
                        {input_prompt}

                        User Question: {user_question}
                        """
                        response = model.generate_content([prompt, {"text": context}])
                        st.write("🤖 Assistant:", response.text)

                    elif st.session_state.current_mode == "ocr":
                        # Use the OCR response stored in session state
                        if 'ocr_response' in st.session_state:
                            response = get_gemini_response(input_prompt, st.session_state.ocr_response, user_question)
                            st.write("🤖 Assistant:", response)
                        else:
                            st.error("No OCR response found. Please process the OCR PDF first.")
                except Exception as e:
                    st.error(f"Error generating response: {str(e)}")
        else:
            st.warning("Please enter a question.")

# Footer with custom background color and styling
st.markdown("""
    <style>
    .footer {
        background-color: #f8f9fa;
        padding: 1.5rem;
        border-radius: 15px;
        margin-top: 2rem;
        text-align: center;
    }
    .footer p {
        font-family: 'Georgia', serif;
        font-size: 1rem;
        color: #666;
        margin: 0;
    }
    </style>
    <div class="footer">
        <p>Built with ❤ by TeamCodeFusion</p>
    </div>
""", unsafe_allow_html=True)