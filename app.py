import streamlit as st 
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
from io import BytesIO

# Load environment
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Initialize chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Read text from PDFs
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        if pdf is not None:
            pdf_reader = PdfReader(BytesIO(pdf.read()))
            for page in pdf_reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text+= page_text

    return text

# Split text into chunks
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

# Create vector store
def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

# Load Gemini QA chain
def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, 
    if the answer is not in provided context just say, "answer is not available in the context", don't provide the wrong answer

    Context: \n{context}\n
    Question: \n{question}\n
    Answer:
    """
    model = ChatGoogleGenerativeAI(model="models/gemini-1.5-pro-latest", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

# Handle user input and respond
def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)

    chain = get_conversational_chain()

    response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)

    # Save to history
    st.session_state.chat_history.append(("user", user_question))
    st.session_state.chat_history.append(("assistant", response["output_text"]))

# Main app logic
def main():
    st.set_page_config(page_title="Chat With Multiple PDF", layout="centered")
    st.title("📚 Chat with PDF using Gemini Pro")

    # Chat history display (chat-style UI)
    for role, message in st.session_state.chat_history:
        with st.chat_message(role):
            st.markdown(message)

    # User input field
    user_question = st.chat_input("Ask a question about your PDF...")
    if user_question:
        with st.chat_message("user"):
            st.markdown(user_question)
        user_input(user_question)
        with st.chat_message("assistant"):
            st.markdown(st.session_state.chat_history[-1][1])  # show last bot response

    # Sidebar
    with st.sidebar:
        st.header("📎 Upload PDFs")
        pdf_docs = st.file_uploader("Upload your PDF files", accept_multiple_files=True)
        if st.button("🔍 Submit & Process"):
            if pdf_docs:
                with st.spinner("Processing..."):
                    raw_text = get_pdf_text(pdf_docs)
                    if not raw_text.strip():
                        st.error("❌ No extractable text found in the uploaded PDFs.")

                        return

                    text_chunks = get_text_chunks(raw_text)
                    get_vector_store(text_chunks)
                    st.success("✅ Done! You can now start chatting.")
            else:
                st.warning("Please upload at least one PDF.")

        if st.button("🧹 Clear Chat History"):
            st.session_state.chat_history = []
            st.rerun()

# Run the app
if __name__ == "__main__":
    main()
