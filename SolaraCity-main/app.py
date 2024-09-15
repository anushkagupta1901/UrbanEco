import streamlit as st
import pandas as pd
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from dotenv import load_dotenv
from datetime import datetime
import logging
from collections import deque

logging.basicConfig(
    filename='app.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

load_dotenv()
google_api_key = os.getenv("GOOGLE_API_KEY")

if not google_api_key:
    logging.error("Google API key not found in environment variables.")
else:
    genai.configure(api_key=google_api_key)

# Function to load PDF from backend instead of user-upload
def load_backend_pdf():
    backend_pdf_path = "SolaraCity ChatBot.pdf"  # Update this to your actual PDF file path
    try:
        pdf_reader = PdfReader(backend_pdf_path)
        text_pages = []
        for i, page in enumerate(pdf_reader.pages):
            text = page.extract_text()
            if text:
                text_pages.append((text, i + 1))
        return text_pages
    except Exception as e:
        logging.error(f"Error reading backend PDF file: {e}")
        return []

def get_text_chunks(text_pages):
    try:
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
        chunks = []
        for text, page_num in text_pages:
            split_chunks = text_splitter.split_text(text)
            for chunk in split_chunks:
                chunks.append((chunk, page_num))
        return chunks
    except Exception as e:
        logging.error(f"Error splitting text into chunks: {e}")
        return []

def get_vector_store(text_chunks):
    try:
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        texts = [chunk for chunk, _ in text_chunks]
        metadatas = [{"page_num": page_num} for _, page_num in text_chunks]
        vector_store = FAISS.from_texts(texts, embedding=embeddings, metadatas=metadatas)
        vector_store.save_local("faiss_index")
    except Exception as e:
        logging.error(f"Error creating vector store: {e}")

def get_conversational_chain():
    try:
        prompt_template = """
        You are the solaracity chatbot and a great explainer 
        Answer in that format . You have to explain in a way so that the person understands very well the concept and is clear regarding the same.
        History:\n{memory}
        Context:\n {context}?\n
        Question: \n{question}\n
        Answer in as detail as possible

        Answer:
        """
        model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
        prompt = PromptTemplate(template=prompt_template, input_variables=["memory", "context", "question"])
        chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
        return chain
    except Exception as e:
        logging.error(f"Error setting up conversational chain: {e}")
        return None

def user_input(user_question, memory):
    try:
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
        docs = new_db.similarity_search(user_question)

        chain = get_conversational_chain()
        if chain:
            response = chain({"memory": memory, "input_documents": docs, "question": user_question}, return_only_outputs=True)

            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            st.write("Reply: ", response["output_text"])
            
            return timestamp, response['output_text']
        else:
            return None, None
    except Exception as e:
        logging.error(f"Error processing user input: {e}")
        return None, None

# Read last lines from log file
def read_last_lines(filename, lines_count):
    with open(filename, 'r') as file:
        return ''.join(deque(file, maxlen=lines_count))

# Function to initialize the backend PDF processing
def initialize_pdf_processing():
    text_pages = load_backend_pdf()
    if text_pages:
        text_chunks = get_text_chunks(text_pages)
        if text_chunks:
            get_vector_store(text_chunks)

def main():
    st.set_page_config(page_title="SolaraCity ChatBot")
    st.header("SolaraCity ChatBot")

    if 'chat_history' not in st.session_state:
        st.session_state['chat_history'] = []

    # Simple input interface for chatbot
    user_question = st.text_input("Ask a Question about the document")

    if user_question:
        timestamp, ai_response = user_input(user_question, st.session_state['chat_history'])
        if timestamp and ai_response:
            st.session_state['chat_history'].append(("Time", timestamp))
            st.session_state['chat_history'].append(("USER", user_question))
            st.session_state['chat_history'].append(("SolaraCity Bot", ai_response))

    if st.session_state['chat_history']:
        st.title("Chat History")
        for role, text in st.session_state['chat_history']:
            st.write(f"`{role}`: {text}")
        st.write("-----")

    # Display logs
    with st.sidebar:
        if st.toggle("Show Logs"):
            last_lines = read_last_lines("app.log", 5)
            st.text(last_lines)

if __name__ == "__main__":
    initialize_pdf_processing()  # Initialize backend PDF processing at startup
    main()
