import streamlit as st
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_ollama import OllamaEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM
import os
from fpdf import FPDF
from langchain_core.prompts import PromptTemplate

# UI Customization
st.set_page_config(page_title="📘 DocuMind AI", layout="wide")

# Embedding and LLM Setup
EMBEDDING_MODEL = OllamaEmbeddings(model="deepseek-r1:1.5b")
DOCUMENT_VECTOR_DB = InMemoryVectorStore(EMBEDDING_MODEL)
LANGUAGE_MODEL = OllamaLLM(model="deepseek-r1:1.5b")

# Paths
PDF_STORAGE_PATH = 'document_store/pdfs/'
os.makedirs(PDF_STORAGE_PATH, exist_ok=True)

# Functions
def save_uploaded_files(uploaded_files):
    paths = []
    for uploaded_file in uploaded_files:
        file_path = os.path.join(PDF_STORAGE_PATH, uploaded_file.name)
        with open(file_path, "wb") as file:
            file.write(uploaded_file.getbuffer())
        paths.append(file_path)
    return paths

def load_pdf_documents(file_paths):
    docs = []
    for path in file_paths:
        loader = PDFPlumberLoader(path)
        docs.extend(loader.load())
    return docs

def chunk_documents(raw_documents):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200, add_start_index=True)
    return splitter.split_documents(raw_documents)

def index_documents(document_chunks):
    DOCUMENT_VECTOR_DB.add_documents(document_chunks)

def find_related_documents(query):
    return DOCUMENT_VECTOR_DB.similarity_search(query)

def generate_answer(user_query, context_documents):
    context_text = "\n\n".join([doc.page_content for doc in context_documents])
    prompt_template = ChatPromptTemplate.from_template("""
        You are an expert research assistant. Use the provided context to answer the query concisely.
        Query: {user_query}
        Context: {document_context}
        Answer:
    """)
    response_chain = prompt_template | LANGUAGE_MODEL
    return response_chain.invoke({"user_query": user_query, "document_context": context_text})

def summarize_document(document_text):
    summary_prompt = PromptTemplate(template="Summarize the following document briefly:\n{text}\nSummary:")
    summary_chain = summary_prompt | LANGUAGE_MODEL
    return summary_chain.invoke({"text": document_text})

def export_to_pdf(text, filename="summary.pdf"):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.multi_cell(0, 10, text)
    file_path = os.path.join(PDF_STORAGE_PATH, filename)
    pdf.output(file_path)
    return file_path

# UI
st.title("📘 DocuMind AI")
st.markdown("### Your Intelligent Document Assistant")
st.markdown("---")

# File Upload
uploaded_files = st.file_uploader("Upload PDFs", type=["pdf"], accept_multiple_files=True)

if uploaded_files:
    saved_paths = save_uploaded_files(uploaded_files)
    raw_docs = load_pdf_documents(saved_paths)
    processed_chunks = chunk_documents(raw_docs)
    index_documents(processed_chunks)

    st.success("✅ Files processed successfully! Ask your questions below.")

    # User Input
    user_input = st.chat_input("Ask about your documents...")

    if user_input:
        with st.spinner("Analyzing..."):
            relevant_docs = find_related_documents(user_input)
            ai_response = generate_answer(user_input, relevant_docs)
        
        st.write(ai_response)

        # Export Option
        if st.button("Export Response to PDF"):
            pdf_path = export_to_pdf(ai_response)
            st.success(f"Exported to PDF: {pdf_path}")
            with open(pdf_path, "rb") as f:
                st.download_button("Download PDF", f, file_name="summary.pdf")

    # Summarization
    if st.button("Summarize All Documents"):
        combined_text = " ".join([doc.page_content for doc in processed_chunks])
        summary = summarize_document(combined_text)
        st.write(summary)
        if st.button("Export Summary to PDF"):
            summary_path = export_to_pdf(summary, "document_summary.pdf")
            with open(summary_path, "rb") as f:
                st.download_button("Download Summary", f, file_name="document_summary.pdf")
