import os
import streamlit as st
from langchain_groq.chat_models import ChatGroq
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.retrieval import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_google_genai.embeddings import GoogleGenerativeAIEmbeddings
from PyPDF2 import PdfReader
from dotenv import load_dotenv

load_dotenv()

# Load the GROQ and Embeddings
groq_api_key = os.getenv("GROQ_API_KEY")
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

st.title("Chat with Document Q&A")

llm = ChatGroq(groq_api_key=groq_api_key, model_name="llama3-8b-8192")


#Prompt Template
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

# uploade the pdf
uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])

# Vector Embedding
def vector_embedding(pdf_file):
    st.session_state.embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    pdf_reader = PdfReader(pdf_file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()

    st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    st.session_state.final_documents = st.session_state.text_splitter.split_text(text)
    st.session_state.vectors = FAISS.from_texts(st.session_state.final_documents, st.session_state.embeddings)

if uploaded_file and st.button("Load Document"):
    vector_embedding(uploaded_file)

# Initialize chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

prompt1 = st.text_input("Ask a question from the documents")

if st.button("Send"):
    if prompt1 and "vectors" in st.session_state:
        document_chain = create_stuff_documents_chain(llm, prompt)
        retriever = st.session_state.vectors.as_retriever()
        retrieval_chain = create_retrieval_chain(retriever, document_chain)

        response = retrieval_chain.invoke({'input': prompt1})

       
        st.session_state.chat_history.append({"question": prompt1, "response": response['answer']})


for chat in st.session_state.chat_history:

    # User Response
    st.markdown(f"""
    <div style="display: flex; justify-content: flex-end;">
        <div style="background-color: #e1ffc7; color: black;padding: 10px; border-radius: 15px; margin: 5px; max-width: 70%;">
            <strong>You:</strong> {chat['question']}
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # AI's response 
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
