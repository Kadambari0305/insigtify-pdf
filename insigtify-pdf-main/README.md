# ⚡ Insigtify PDF — Multi-Modal Document & Photo Analysis Hub

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://insigtify-pdf-fnmamgvrcsryr5qmkfv53w.streamlit.app/)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Powered by Gemini](https://img.shields.io/badge/AI-Google%20Gemini%203.6%20Flash-4285F4.svg)](https://aistudio.google.com/)

**Insigtify PDF** is an intelligent multi-modal document research assistant built with Streamlit, LangChain, FAISS, and Google Gemini AI. It enables users to upload, extract, index, and query information across multiple sources simultaneously—including standard printed PDFs, scanned handwritten documents, receipts, invoices, and photo images.

---

## 🌟 Key Features

- **📄 Multi-Format File Support**: Upload and analyze multiple PDF documents and images (`.pdf`, `.png`, `.jpg`, `.jpeg`, `.webp`) at once.
- **👁️ Automated Vision OCR**: Automatically detects scanned/image-only PDFs and photos, employing Gemini Vision OCR for extraction of text, tables, and handwritten notes.
- **🔍 Vector RAG Search**: Uses `LangChain` and `FAISS` with Google Generative AI embeddings (`models/gemini-embedding-001`) to index chunks and retrieve contextual facts.
- **💬 Grounded AI Responses**: Answers complex questions across multiple files with exact source citations (e.g. `[Source: document.pdf]`).
- **🔐 User Authentication & Session History**: Built-in user registration & login (with SHA-256 hashed passwords), session saving, and past chat history switching backed by **MongoDB Cloud** (with automatic local **SQLite fallback**).
- **🖼️ Photo & PDF Inspector**: Preview uploaded images and inspect processed documents directly in the UI.

---

## 🏗️ Architecture & Tech Stack

- **Frontend & Framework**: [Streamlit](https://streamlit.io/)
- **LLM & OCR Engine**: Google Gemini (`gemini-3.6-flash`) via `google-generativeai`
- **Embeddings**: `GoogleGenerativeAIEmbeddings` (`models/gemini-embedding-001`)
- **Vector Database**: [FAISS](https://github.com/facebookresearch/faiss) via `langchain-community`
- **Text Processing & Chunking**: `langchain-text-splitters` & `pypdf`
- **Database**: MongoDB Cloud (`pymongo`) with local SQLite fallback (`sqlite3`)
- **Environment & Config**: `python-dotenv`

---

## 🚀 Live Demo

Access the deployed application on Streamlit Community Cloud:
👉 **[Insigtify PDF Live App](https://insigtify-pdf-fnmamgvrcsryr5qmkfv53w.streamlit.app/)**

---

## 🛠️ Installation & Local Setup

### Prerequisites
- Python 3.10 or higher
- A free **Google Gemini API Key** from [Google AI Studio](https://aistudio.google.com/app/apikey)

### 1. Clone the Repository
```bash
git clone https://github.com/Kadambari0305/insigtify-pdf.git
cd insigtify-pdf
```

### 2. Create & Activate Virtual Environment
```bash
# On Windows
python -m venv venv
venv\Scripts\activate

# On macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables
Create a `.env` file in the root directory:
```env
GOOGLE_API_KEY=YOUR_GEMINI_API_KEY_HERE
MONGO_DB=document_analysis
MONGO_URI=mongodb+srv://<username>:<password>@cluster0.example.mongodb.net/pdf_chatbot
```
> *Note: If `MONGO_URI` is omitted or unavailable, the application automatically falls back to a local SQLite database in `./data/app_database.db`.*

### 5. Run the Streamlit Application
```bash
streamlit run app.py
```

Open `http://localhost:8501` in your browser.

---

## ☁️ Deployment Guide (Streamlit Community Cloud)

1. Push your code to GitHub (ensure `.env` is listed in `.gitignore`).
2. Go to [share.streamlit.io](https://share.streamlit.io) and create a new app pointing to your repository branch `main` and main file `app.py`.
3. In **App Settings $\rightarrow$ Secrets**, add your secrets in valid TOML format:
   ```toml
   GOOGLE_API_KEY = "AIzaSy..."
   MONGO_DB = "document_analysis"
   MONGO_URI = "mongodb+srv://username:password@cluster.mongodb.net/pdf_chatbot"
   ```
4. Click **Save** and deploy!

---

## 📁 Repository Structure

```
├── app.py              # Main Streamlit Web Application UI & RAG logic
├── db_manager.py       # User Authentication & Database Manager (MongoDB + SQLite fallback)
├── requirements.txt    # Project dependencies
├── .gitignore          # Git ignore rules for credentials & cached data
└── README.md           # Documentation
```

---

## 🛡️ License

This project is open-source and available under the [MIT License](LICENSE).
