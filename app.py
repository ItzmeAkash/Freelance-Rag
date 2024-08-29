import os
import streamlit as st
from langchain_groq.chat_models import ChatGroq
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.retrieval import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_google_genai.embeddings import GoogleGenerativeAIEmbeddings
from PyPDF2 import PdfReader
from dotenv import load_dotenv

load_dotenv()

# Load the GROQ and Embeddings
groq_api_key = os.getenv("GROQ_API_KEY")
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

st.title("Document Q&A")

llm = ChatGroq(groq_api_key=groq_api_key, model_name="llama3-8b-8192")

prompt = ChatPromptTemplate.from_template(
    """
    Answer the questions based on the provided context only.
    Please provide the most accurate response based on the question.
    <context>
    {context}
    <context>
    Questions: {input}
    """
)

def vector_embedding():
    if "vectors" not in st.session_state:
        st.session_state.embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        st.session_state.loader = PyPDFDirectoryLoader("./data")  # Data Ingestion
        st.session_state.docs = st.session_state.loader.load()  # Document Loading
        st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)  # Chunk Creation
        st.session_state.final_documents = st.session_state.text_splitter.split_documents(st.session_state.docs[:20])  # Splitting
        st.session_state.vectors = FAISS.from_documents(st.session_state.final_documents, st.session_state.embeddings)  # Vector embeddings

st.button("Load Documents", on_click=vector_embedding)

# Initialize chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

prompt1 = st.text_input("Ask a question from the documents")

if st.button("Send"):
    if prompt1:
        document_chain = create_stuff_documents_chain(llm, prompt)
        retriever = st.session_state.vectors.as_retriever()
        retrieval_chain = create_retrieval_chain(retriever, document_chain)

        response = retrieval_chain.invoke({'input': prompt1})

        # Store the question and response in chat history
        st.session_state.chat_history.append({"question": prompt1, "response": response['answer']})

# Display chat history with custom styling for a chatbot layout
for chat in st.session_state.chat_history:
    # User's question (right side)
    st.markdown(f"""
    <div style="display: flex; justify-content: flex-end;">
        <div style="background-color: #e1ffc7; color: black;padding: 10px; border-radius: 15px; margin: 5px; max-width: 70%;">
            <strong>You:</strong> {chat['question']}
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # AI's response (left side)
    st.markdown(f"""
    <div style="display: flex; justify-content: flex-start;">
        <div style="background-color: #f1f0f0; color: black; padding: 10px; border-radius: 15px; margin: 5px; max-width: 70%;">
            <strong>Ai Response:</strong> {chat['response']}
        </div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("""
    <style>
        .stTextInput > div > div > input {
            border-radius: 15px;
            padding: 10px;
        }
    </style>
""", unsafe_allow_html=True)
