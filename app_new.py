import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain_community.document_loaders import WebBaseLoader
# Old import
#from langchain.embeddings import OllamaEmbeddings

# New import
#from langchain_community.embeddings import OllamaEmbeddings
from langchain.embeddings import HuggingFaceEmbeddings  # Using HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document
import time
import requests
from bs4 import BeautifulSoup

from dotenv import load_dotenv
load_dotenv()

## load the Groq API key
groq_api_key=os.environ['GROQ_API_KEY']

if "vector" not in st.session_state:
    #st.session_state.embeddings=OllamaEmbeddings()
    st.session_state.embeddings = HuggingFaceEmbeddings()
    
    # Fetch the webpage content using requests and parse it with BeautifulSoup
    url = "https://www.shankara.com/pages/blog"
    response = requests.get(url)
    if response.status_code == 200:
        html_content = response.text
        soup = BeautifulSoup(html_content, "html.parser")
        
        # Extract text from relevant elements, like paragraphs or headings
        texts = [p.get_text() for p in soup.find_all("p")]
        # Create a list of Document objects
        st.session_state.final_documents = [Document(page_content=text) for text in texts]        # Create FAISS vectors from documents and embeddings
        st.session_state.vectors = FAISS.from_documents(st.session_state.final_documents, st.session_state.embeddings)
    else:
        st.error(f"Failed to fetch content from {url}")


st.title("ChatGroq Demo")
llm=ChatGroq(groq_api_key=groq_api_key,
             model_name="Llama3-8b-8192")

prompt=ChatPromptTemplate.from_template(
"""
Answer the questions based on the provided context only.
Please provide the most accurate response based on the question
<context>
{context}
<context>
Questions:{input}

"""
)
document_chain = create_stuff_documents_chain(llm, prompt)
retriever = st.session_state.vectors.as_retriever()
retrieval_chain = create_retrieval_chain(retriever, document_chain)

prompt=st.text_input("Input you prompt here")

if prompt:
    start=time.process_time()
    response=retrieval_chain.invoke({"input":prompt})
    print("Response time :",time.process_time()-start)
    st.write(response['answer'])

    # With a streamlit expander
    with st.expander("Document Similarity Search"):
        # Find the relevant chunks
        for i, doc in enumerate(response["context"]):
            st.write(doc.page_content)
            st.write("--------------------------------")
