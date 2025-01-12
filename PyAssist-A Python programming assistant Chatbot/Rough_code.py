
## THIS IS UNSTRUCTURED ROUGH CODE .



import pandas as pd


knowledge_base = pd.read_csv('Dataset_Python_Question_Answer.csv')

print(knowledge_base.head())

from langchain.docstore.document import Document

documents = [
    Document(page_content=row['Question'] + "\n" + row['Answer'])
    for index, row in knowledge_base.iterrows()  
]

print(documents[0].page_content)

from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-MiniLM-L6-v2')

embeddings = [model.encode(doc.page_content) for doc in documents]

print(embeddings[0])

# !pip install faiss-cpu

from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings

embedding_model = OpenAIEmbeddings(openai_api_key="sk-svcacct-HJ6RhrBpdhvvSIb_PAt2kZmN9or5jnd4Pi4I_xh9RsY8k8trcPCQy6hwKiOzjvHvXM9T3BlbkFJDT5GBdiJbGs2HHJbayU7T9c-hVFt17tMeSOzMti6pKX1cUyHc6jg_QSXqPWwoTAylAA")

vector_store = FAISS.from_documents(documents, embedding_model)

from langchain.llms import OpenAI
from langchain.chains import RetrievalQA

llm = OpenAI(openai_api_key="sk-svcacct-HJ6RhrBpdhvvSIb_PAt2kZmN9or5jnd4Pi4I_xh9RsY8k8trcPCQy6hwKiOzjvHvXM9T3BlbkFJDT5GBdiJbGs2HHJbayU7T9c-hVFt17tMeSOzMti6pKX1cUyHc6jg_QSXqPWwoTAylAA")

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="map_reduce",  
    retriever=vector_store.as_retriever()  
)




question = "what is list?"  
response = qa_chain.run(question)

print("Response from Chatbot:", response)
