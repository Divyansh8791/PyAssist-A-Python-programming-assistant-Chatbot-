# Import dependencies from Backend.
from backend import load_dataset , create_vector_store , create_qa_chain
import streamlit as st
import pandas as pd

# Creating the frontend.
def main():
        st.title("PyAssist : A Python programming assistant Chatbot.")
        st.write("Ask questions based on the python basics.") 

        with st.spinner("Processing the dataset..."):
            documents = load_dataset()
            vector_store = create_vector_store()
            qa_chain = create_qa_chain()
            st.success("Knowledge base loaded and chatbot is ready!")

        # Chatbot Section
        st.header("Chat with the Bot")
        question = st.text_input("Ask a question:")
        if question:
            with st.spinner("Fetching response..."):
                response = qa_chain.run(question)
            st.write("**Response:**", response)

        # Footer
        st.sidebar.title("About")
        st.sidebar.info(
            "This is a Retrieval-Augmented Generation (RAG) chatbot built with LangChain, "
            "FAISS, OpenAI, and Streamlit."
        )


if __name__ == "__main__":
    main()
