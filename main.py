from dotenv import load_dotenv
import os
import sys
import time
import streamlit as st
from langchain_openai import OpenAI
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains.qa_with_sources.retrieval import RetrievalQAWithSourcesChain

load_dotenv()

# Access OpenAI API through key
key = os.getenv('OPENAI_API_KEY')
os.environ['OPENAI_API_KEY'] = key

# Set the title of the page
st.header('Data Retrieval Tool 🔎', divider='violet')

# Placeholder for the url process
url_process_placeholder = st.empty()

# Create the sidebar
st.sidebar.title('Website URLs')

urls = []
for num in range(1, 6):
    url = st.sidebar.text_input(f'URL {num}')
    urls.append(url)

isProcessed = st.sidebar.button('Retrieve Information')

# Check if the url list is all empty
urls_not_completely_empty = any(True if len(url) else False for url in urls)

if isProcessed and urls_not_completely_empty == True:
    # Load the information sources
    url_process_placeholder.text('Loading the data...✅✅✅')
    loader = UnstructuredURLLoader(urls=urls)
    data = loader.load()
    # If the sources are unable to be loaded, then stop the program
    if not data:
        st.error('Unable to load the source information', icon='🚨')
        sys.exit()
    time.sleep(2)
    # Split the contents
    url_process_placeholder.text('Splitting the data...✅✅✅')
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=0
    )
    docs = splitter.split_documents(data)
    time.sleep(2)
    # Embed all the texts and store them in FAISS
    url_process_placeholder.text('Embedding the data...✅✅✅')
    embeddings = OpenAIEmbeddings()
    db = FAISS.from_documents(docs, embeddings)
    time.sleep(2)
    # Save database to a file
    db.save_local('./database/faiss_index')
    url_process_placeholder.text('All Done...✅✅✅')
elif isProcessed and urls_not_completely_empty == False:
    st.warning('Please provide at least 1 source of information', icon="⚠️")

# Create a text input for writing question
question = st.text_input('Question:')
# Placeholder for data retrieving
data_retrieving_placeholder = st.empty()
if question:
    # Call database
    embeddings = OpenAIEmbeddings()
    db = FAISS.load_local('./database/faiss_index', embeddings)
    # Call LLM
    data_retrieving_placeholder.text('Calling LLM...✅✅✅')
    llm = OpenAI(temperature=0.9, max_tokens=500)
    # Setup the retriever
    chain = RetrievalQAWithSourcesChain.from_llm(llm=llm, retriever=db.as_retriever())
    # Return the answer
    result = chain({'question': question}, return_only_outputs=True)
    st.header('Answer:')
    container = st.container(border=True)
    container.write(result['answer'])