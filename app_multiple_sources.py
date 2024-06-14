import streamlit as st
import os
import time
import pprint
from dotenv import load_dotenv
from langchain.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.agents import create_openai_tools_agent, AgentExecutor
from langchain_community.tools import WikipediaQueryRun, ArxivQueryRun
from langchain_community.utilities import WikipediaAPIWrapper, ArxivAPIWrapper

# Load environment variables
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

# Initialize Streamlit session state for the vector store
if "vector" not in st.session_state:
    # Set up embeddings and loaders
    st.session_state.embeddings = OpenAIEmbeddings(api_key=openai_api_key)
    st.session_state.loader = WebBaseLoader("https://docs.smith.langchain.com/")
    st.session_state.docs = st.session_state.loader.load()

    # Split documents
    st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    st.session_state.final_documents = st.session_state.text_splitter.split_documents(st.session_state.docs[:50])

    # Create FAISS vector store
    st.session_state.vectors = FAISS.from_documents(st.session_state.final_documents, st.session_state.embeddings)

# Set up Wikipedia and Arxiv tools
api_wrapper = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=200)
wiki = WikipediaQueryRun(api_wrapper=api_wrapper)

arxiv_wrapper = ArxivAPIWrapper(top_k_results=1, doc_content_chars_max=200)
arxiv = ArxivQueryRun(api_wrapper=arxiv_wrapper)

# Streamlit app title
st.title("RAG Q&A App")

# Instantiate the LLM
llm = OpenAI(api_key=openai_api_key, model="gpt-4")

# Load the prompt template
prompt_template = PromptTemplate("hwchase17/openai-functions-agent")

# Create the retriever tool
retriever_tool = st.session_state.vectors.as_retriever()

# Create the QA chain
qa_chain = RetrievalQA.from_chain_type(
    llm,
    retriever=retriever_tool,
    chain_type_kwargs={"prompt": prompt_template},
)

# Create the OpenAI tools agent
tools = [wiki, arxiv, retriever_tool]
agent = create_openai_tools_agent(llm, tools, prompt_template)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# Input prompt from user
question = st.text_input("Input your prompt here")

# Process the question if provided
if question:
    # Measure response time
    start = time.process_time()
    response = agent_executor.invoke({"input": question})
    response_time = time.process_time() - start

    # Display response time and answer
    st.write("Response time:", response_time)
    st.write(response['answer'])

    # Pretty print the result
    pp = pprint.PrettyPrinter(indent=5)
    answer = pp.pprint(response["result"])

    # Document similarity search results in expander
    with st.expander("Document Similarity Search"):
        for i, doc in enumerate(response["context"]):
            st.write(doc.page_content)
            st.write("--------------------------------")
