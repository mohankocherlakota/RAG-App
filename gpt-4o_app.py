import streamlit as st
import os
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import RetrievalQA
from langchain_objectbox.vectorstores import ObjectBox
from langchain.embeddings import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain import hub
import time
import pprint

from dotenv import load_dotenv
load_dotenv()

# Load the OpenAI API key
openai_api_key = os.getenv('OPENAI_API_KEY')

# Check if the necessary components are already in session state
if "vector" not in st.session_state:
    st.session_state.embeddings = OpenAIEmbeddings(api_key=openai_api_key)
    st.session_state.loader = WebBaseLoader("https://docs.smith.langchain.com/")
    st.session_state.docs = st.session_state.loader.load()
    
    st.session_state.text_splitter = RecursiveCharacterTextSplitter()
    st.session_state.final_documents = st.session_state.text_splitter.split_documents(st.session_state.docs[:50])
    st.session_state.vector = ObjectBox.from_documents(
        st.session_state.final_documents, 
        st.session_state.embeddings,
        embedding_dimensions=768)

# Streamlit app title
st.title("GPT-4 RAG App")

# Instantiate the LLM
llm = OpenAI(api_key=openai_api_key, model="gpt-4o")

# Load the prompt template
prompt_template = hub.pull("rlm/rag-prompt")

# Create the QA chain
qa_chain = RetrievalQA.from_chain_type(
    llm,
    retriever=st.session_state.vector.as_retriever(),
    chain_type_kwargs={"prompt": prompt_template}
)

# Input prompt from user
question = st.text_input("Input your prompt here")

# Process the question if provided
if question:
    result = qa_chain({"query": question})
    
    # Pretty print the result
    pp = pprint.PrettyPrinter(indent=5)
    answer = pp.pprint(result["result"])

    # Measure response time
    start = time.process_time()
    response = qa_chain.invoke({"input": question})
    response_time = time.process_time() - start
    st.write("Response time:", response_time)
    st.write(response['answer'])

    # Document similarity search results in expander
    with st.expander("Document Similarity Search"):
        for i, doc in enumerate(response["context"]):
            st.write(doc.page_content)
            st.write("--------------------------------")
