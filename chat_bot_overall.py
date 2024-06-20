import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.vectorstores import FAISS
from langchain_community.vectorstores import FAISS as CommunityFAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import os

load_dotenv()
os.getenv("GOOGLE_API_KEY")
import google.generativeai as genai
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

def process_pdfs(pdf_docs):
    raw_text = get_pdf_text(pdf_docs)
    text_chunks = get_text_chunks(raw_text)
    get_vector_store(text_chunks)

def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
    provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
    Context:\n {context}?\n
    Question: \n{question}\n
    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    try:
        new_db = CommunityFAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
        docs = new_db.similarity_search(user_question)
        chain = get_conversational_chain()
        response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
        return response
    except Exception as e:
        print(f"Error during processing: {e}")
        return {"error": "An error occurred while processing the question."}

def main():
    st.set_page_config("Chat PDF")
    st.header("Chat with Manual - Knorr Bremse")

    # Initialize empty chat history in session state
    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []

    # Hardcoded list of PDF files
    pdf_files = ["manual1.pdf"]  # Add your PDF file paths here

    # Check if the vector store exists, if not, process the PDFs
    if not os.path.exists("faiss_index"):
        with st.spinner("Processing PDFs..."):
            process_pdfs(pdf_files)
        st.success("PDFs processed successfully. You can now ask questions.")

    # Create a horizontal container for input field and button
    col1, col2 = st.columns([4, 1])
    user_question = col1.text_input("Ask a Question from the PDF Files", key="user_question")

    # Add custom CSS to adjust the button position
    st.markdown("""
        <style>
        .stButton button {
            margin-top: 28px;
            margin-left: 10px;
        }
        </style>
    """, unsafe_allow_html=True)

    if col2.button("Submit"):
        if user_question:
            response = user_input(user_question)  # Process the question using the chatbot

            if isinstance(response, dict) and "error" in response:
                st.error(response["error"])  # Display error message
            else:
                # Append user question and response to chat history
                answer_text = response["output_text"]
                st.session_state["chat_history"].append({"question": user_question, "answer": answer_text})

    # Reverse chat history to display latest question at the top
    chat_history = st.session_state["chat_history"][::-1]

    # Display chat history with styled boxes
    if chat_history:
        st.subheader("Chat History:")

        for entry in chat_history:
            with st.container():
                # Question box with light red background and full question text inside
                st.write(
                    f'<div style="background-color: #9FE2BF; padding: 10px; margin: 5px;"><b>Question:</b> {entry["question"]}</div>',
                    unsafe_allow_html=True,
                )
                # Answer box with light blue background using HTML
                answer_text = entry["answer"]
                st.write(
                    f'<div style="background-color: lightblue; padding: 10px; margin: 5px;"><b>Answer:</b> {answer_text}</div>',
                    unsafe_allow_html=True,
                )

if __name__ == "__main__":
    main()
