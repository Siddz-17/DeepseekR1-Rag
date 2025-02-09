import streamlit as st
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_ollama import OllamaEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM
import os  # Import os module to handle directory operations
from PIL import Image  # Import PIL for image processing
import speech_recognition as sr  # Import for audio processing
import pytesseract  # Import pytesseract for OCR
import easyocr  # Import easyocr for OCR
import numpy as np  # Import numpy for array manipulation

st.markdown("""
    <style>
    .stApp {
        background-color: #0E1117;
        color: #FFFFFF;
    }
    
    /* Chat Input Styling */
    .stChatInput input {
        background-color: #1E1E1E !important;
        color: #FFFFFF !important;
        border: 1px solid #3A3A3A !important;
    }
    
    /* User Message Styling */
    .stChatMessage[data-testid="stChatMessage"]:nth-child(odd) {
        background-color: #1E1E1E !important;
        border: 1px solid #3A3A3A !important;
        color: #E0E0E0 !important;
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
    }
    
    /* Assistant Message Styling */
    .stChatMessage[data-testid="stChatMessage"]:nth-child(even) {
        background-color: #2A2A2A !important;
        border: 1px solid #404040 !important;
        color: #F0F0F0 !important;
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
    }
    
    /* Avatar Styling */
    .stChatMessage .avatar {
        background-color: #00FFAA !important;
        color: #000000 !important;
    }
    
    /* Text Color Fix */
    .stChatMessage p, .stChatMessage div {
        color: #FFFFFF !important;
    }
    
    .stFileUploader {
        background-color: #1E1E1E;
        border: 1px solid #3A3A3A;
        border-radius: 5px;
        padding: 15px;
    }
    
    h1, h2, h3 {
        color: #00FFAA !important;
    }
    </style>
    """, unsafe_allow_html=True)

PROMPT_TEMPLATE = """
You are an expert research assistant. Use the provided context to answer the query. 
If unsure, state that you don't know. Be concise and factual (max 3 sentences).

Query: {user_query} 
Context: {document_context} 
Answer:
"""
PDF_STORAGE_PATH = 'document_store/pdfs/'
EMBEDDING_MODEL = OllamaEmbeddings(model="deepseek-r1:1.5b")
DOCUMENT_VECTOR_DB = InMemoryVectorStore(EMBEDDING_MODEL)
LANGUAGE_MODEL = OllamaLLM(model="deepseek-r1:1.5b")

# Initialize the EasyOCR reader
reader = easyocr.Reader(['en'])  # Specify the language(s) you want to use

# Initialize a variable to store extracted text
extracted_text = ""

def save_uploaded_file(uploaded_file):
    # Ensure the directory exists
    os.makedirs(PDF_STORAGE_PATH, exist_ok=True)  # Create the directory if it doesn't exist
    file_path = PDF_STORAGE_PATH + uploaded_file.name
    with open(file_path, "wb") as file:
        file.write(uploaded_file.getbuffer())
    return file_path

def load_pdf_documents(file_path):
    document_loader = PDFPlumberLoader(file_path)
    return document_loader.load()

def chunk_documents(raw_documents):
    text_processor = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        add_start_index=True
    )
    return text_processor.split_documents(raw_documents)

def index_documents(document_chunks):
    DOCUMENT_VECTOR_DB.add_documents(document_chunks)

def find_related_documents(query):
    return DOCUMENT_VECTOR_DB.similarity_search(query)

def generate_answer(user_query, context_documents):
    context_text = "\n\n".join([doc.page_content for doc in context_documents])
    conversation_prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    response_chain = conversation_prompt | LANGUAGE_MODEL
    return response_chain.invoke({"user_query": user_query, "document_context": context_text})


# UI Configuration


st.title("📘 DocuMind AI")
st.markdown("### Your Intelligent Document Assistant")
st.markdown("---")

# File Upload Section
uploaded_file = st.file_uploader(
    "Upload Research Document (PDF) or Image (JPEG/PNG)",
    type=["pdf", "jpg", "jpeg", "png"],
    help="Select a PDF document or an image for analysis",
    accept_multiple_files=False
)

if uploaded_file:
    if uploaded_file.type in ["application/pdf"]:
        saved_path = save_uploaded_file(uploaded_file)
        raw_docs = load_pdf_documents(saved_path)
        processed_chunks = chunk_documents(raw_docs)
        index_documents(processed_chunks)
        
        st.success("✅ Document processed successfully! Ask your questions below.")
        
    elif uploaded_file.type in ["image/jpeg", "image/png"]:
        image = Image.open(uploaded_file)
        # Convert the PIL Image to a NumPy array
        image_np = np.array(image)
        
        # Use easyocr to extract text from the image
        extracted_text = reader.readtext(image_np, detail=0)  # detail=0 returns only the text
        extracted_text = " ".join(extracted_text)  # Join the list into a single string
        
        st.image(image, caption='Uploaded Image', use_column_width=True)
        st.write("Extracted Text: ", extracted_text)  # Display the extracted text
        st.success("✅ Image processed successfully! You can ask questions about it.")
        
    # Add audio processing if needed
    elif uploaded_file.type in ["audio/wav", "audio/mp3"]:
        # Process the audio file
        recognizer = sr.Recognizer()
        with sr.AudioFile(uploaded_file) as source:
            audio_data = recognizer.record(source)
            text = recognizer.recognize_google(audio_data)
            st.write("Transcribed Text: ", text)
            st.success("✅ Audio processed successfully! You can ask questions about it.")
    
    user_input = st.chat_input("Enter your question about the document or image...")
    
    if user_input:
        with st.chat_message("user"):
            st.write(user_input)
        
        with st.spinner("Analyzing..."):
            # Combine the extracted text with any other relevant documents
            relevant_docs = find_related_documents(user_input)  # Modify as needed
            combined_context = [extracted_text] + [doc.page_content for doc in relevant_docs]  # Combine contexts
            
            # Generate the answer using the combined context
            ai_response = generate_answer(user_input, combined_context)
            
        with st.chat_message("assistant", avatar="🤖"):
            st.write(ai_response)
