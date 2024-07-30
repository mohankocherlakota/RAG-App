import streamlit as st
import os
import time
import pprint
from dotenv import load_dotenv
from langchain.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.agents import create_openai_tools_agent, AgentExecutor
from langchain_community.tools import WikipediaQueryRun, ArxivQueryRun
from langchain_community.utilities import WikipediaAPIWrapper, ArxivAPIWrapper
from langchain import hub
from langchain.tools.retriever import create_retriever_tool


# Load environment variables
def load_env_variables():
    load_dotenv()
    return os.getenv("OPENAI_API_KEY")

# Initialize embeddings
def initialize_embeddings(api_key):
    return OpenAIEmbeddings(api_key=api_key)

# Process documents and create FAISS vector store
def create_vector_store(url, embeddings):
    loader = WebBaseLoader(url)
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    final_documents = text_splitter.split_documents(docs[:50])
    vectors = FAISS.from_documents(final_documents, embeddings)
    return vectors

# Initialize tools
def initialize_tools():
    api_wrapper = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=200)
    wiki = WikipediaQueryRun(api_wrapper=api_wrapper)
    arxiv_wrapper = ArxivAPIWrapper(top_k_results=1, doc_content_chars_max=200)
    arxiv = ArxivQueryRun(api_wrapper=arxiv_wrapper)
    return wiki, arxiv

# Create agent executor
def create_agent_executor(llm, tools, prompt_template):
    agent = create_openai_tools_agent(llm, tools, prompt_template)
    return AgentExecutor(agent=agent, tools=tools, verbose=True)

# Main Streamlit app
def main():
    api_key = load_env_variables()
    if not api_key:
        st.error("API key not found. Please set the OPENAI_API_KEY environment variable.")
        return

    if "vector" not in st.session_state:
        embeddings = initialize_embeddings(api_key)
        url = st.text_input("Enter the URL for document loading")
        if url:
            st.session_state.vectors = create_vector_store(url, embeddings)

    if "vectors" not in st.session_state:
        st.warning("Please enter a valid URL to initialize the vector store.")
        return

    wiki, arxiv = initialize_tools()
    llm = ChatOpenAI(api_key=api_key, model="gpt-4")
    prompt_template = hub.pull("hwchase17/openai-functions-agent")
    retriever=st.session_state.vectors.as_retriever()
    retriever_tool=create_retriever_tool(retriever,"Web_Search",
                      "Search for information about Website. For any questions about Company/Person, you must use this tool!")
    tools = [wiki, arxiv, retriever_tool]
    agent_executor = create_agent_executor(llm, tools, prompt_template)

    st.title("RAG Q&A App")
    question = st.text_input("Input your prompt here")

    if question:
        # Measure response time
        start_time = time.process_time()
        response = agent_executor.invoke({"input": question})
        print("Response time :",time.process_time()-start_time)
        st.markdown(response['output'])

if __name__ == "__main__":
    main()
