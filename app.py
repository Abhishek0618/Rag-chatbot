# Import necessary libraries and modules
import streamlit as st                     #Used for building the web app interface.
from PyPDF2 import PdfReader               #For reading text from PDF files.
from langchain.text_splitter import RecursiveCharacterTextSplitter   #For text processing, embeddings, vector storage, and question-answering models.
from langchain_google_genai.embeddings import GoogleGenerativeAIEmbeddings   
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
import os                                                 #For file handling and managing retries.
import time                                               #For file handling and managing retries.
from google.api_core.exceptions import DeadlineExceeded   #For file handling and managing retries.

# Set the Streamlit page configuration
st.set_page_config(page_title="Document Genie", layout="wide")

# Display the title and instructions for the app
st.markdown("""
## Document Genie: Get instant insights from your Documents
## How it works:You'll need a Google API key for the chatbot to access Google's Generative AI models. Obtain your API key https://makersuite.google.com/app/apikey.
1. Enter your API key
2. Upload your documents
3. Ask a question
""")

# Input field for the user to enter their Google API key
api_key = st.text_input("Enter your Google API key:", type="password", key="api_key_input")

# Function to extract text from PDF documents
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

# Function to split text into smaller chunks
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

# Function to create and save a vector store from text chunks
def get_vector_store(text_chunks, api_key, max_retries=3, retry_delay=5):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key)
    
    for attempt in range(max_retries):
        try:
            vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
            vector_store.save_local("faiss_index")
            return vector_store
        except DeadlineExceeded as e:
            st.warning(f"Attempt {attempt + 1} failed: {e}")
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
            else:
                raise e
        except Exception as e:
            st.error(f"An error occurred: {e}")
            raise e

    raise Exception("Exceeded maximum number of retries")

# Function to create a conversational QA chain
def get_conversational_chain(api_key):
    prompt_template = """
    Answer the question as detailed as possible from the provided context. If the answer is not in
    the provided context, just say, "The answer is not available in the context." Do not provide a wrong answer.\n\n
    Context:\n{context}\n
    Question:\n{question}\n
    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3, google_api_key=api_key)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

# Function to process user input and generate a response
def user_input(user_question, api_key):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key)
    new_db = FAISS.load_local("faiss_index", embeddings)
    docs = new_db.similarity_search(user_question)
    chain = get_conversational_chain(api_key)
    response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
    st.write("Reply: ", response["output_text"])

# Main function to run the Streamlit app
def main():
    st.header("AI Document Insight 💁🏻‍♂️")

    # Input field for user questions
    user_question = st.text_input("Ask a question from the PDF files", key="user_question")

    if user_question and api_key:  # Ensure API key and user question are provided
        user_input(user_question, api_key)

    with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader("Upload your PDF files", accept_multiple_files=True, key="pdf_uploader")
        if st.button("Submit & Process", key="process_button") and api_key:  # Check if API key is provided before processing
            with st.spinner("Processing..."):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                try:
                    get_vector_store(text_chunks, api_key)
                    st.success("Processing completed!")
                except Exception as e:
                    st.error(f"Failed to process documents: {e}")

if __name__ == "__main__":
    main()

## For run this in terminal you have to write  ==  streamlit run app.py
