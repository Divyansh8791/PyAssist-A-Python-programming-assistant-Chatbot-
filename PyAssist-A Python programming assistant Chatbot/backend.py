# Import the needed libraries.
import pandas as pd
from langchain.docstore.document import Document
from sentence_transformers import SentenceTransformer
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.llms import OpenAI
from langchain.chains import RetrievalQA
import streamlit as st

@st.cache_data
def load_dataset():
# Loading the dataset.
     knowledge_base = pd.read_csv('Dataset_Python_Question_Answer.csv')
    # Convert each row into a Document.
     documents = [
          Document(page_content=row['Question'] + "\n" + row['Answer'])
          for index, row in knowledge_base.iterrows()  
      ]
     return documents

@st.cache_resource
def create_vector_store():
     documents = load_dataset()
    # Loading a pre-trained model for generating embeddings.
     model = SentenceTransformer('all-MiniLM-L6-v2')

    # Creating embeddings for each document.
     embeddings = [model.encode(doc.page_content) for doc in documents]

    # Define the embeddings model .
    # This API key won't work , please use your own . this is for safety purpose.
     embedding_model = OpenAIEmbeddings(openai_api_key="xh9RsY8k8try6hwKiOzjvHvXM9T3BlbkFJDT5GBdiJbGs2HHJbayU7T9c-hVFt17tMeSOzMti6pKX1cUyHc6jg_QSXqPWwoTAylAA")

    # Creating the FAISS vector store from the documents
     vector_store = FAISS.from_documents(documents, embedding_model)
     return vector_store

@st.cache_resource
def create_qa_chain():
    vector_store = create_vector_store()
    # Initializing the OpenAI LLM with API key.
    llm = OpenAI(openai_api_key="xh9RsY8k8try6hwKiOzjvHvXM9T3BlbkFJDT5GBdiJbGs2HHJbayU7T9c-hVFt17tMeSOzMti6pKX1cUyHc6jg_QSXqPWwoTAylAA")

     # Setting up the RetrievalQA chain.
    qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="map_reduce",  
    retriever=vector_store.as_retriever()  
     )
    return qa_chain
