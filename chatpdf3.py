import streamlit as st
from PyPDF2 import PdfReader, PdfWriter  # For PDF manipulation
import tempfile  # For creating temporary files
import os  # For file management (deletion)
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()
os.getenv("GOOGLE_API_KEY")
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


def user_input(user_question, pdf_docs):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    try:
        new_db = FAISS.load_local("faiss_index", embeddings)
        docs = new_db.similarity_search(user_question)
        chain = get_conversational_chain()
        response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)

        # Find answer text within PDFs
        answer_text = response["output_text"]
        answer_page = None
        answer_pdf = None

        for pdf_index, pdf in enumerate(pdf_docs):
            pdf_reader = PdfReader(pdf)
            for page_num in range(len(pdf_reader.pages)):
                page = pdf_reader.pages[page_num]
                # Check for exact match or keyword match (improve accuracy)
                if answer_text in page.extract_text() or any(word in page.extract_text() for word in answer_text.split()):
                    answer_page = page_num + 1  # Page numbers start from 1
                    answer_pdf = pdf.name
                    break  # Stop iterating within the current PDF

        if answer_page:
            # Update response with PDF name and page number (if found)
            answer_location = f"in document '{answer_pdf}' on Page {answer_page}" if answer_pdf else f"Not from pdf"
            response["answer_location"] = answer_location
        else:
            # Handle case where answer not found in any PDF
            response["answer_location"] = "Answer not found in uploaded PDFs."

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

    user_question = st.text_input("Ask a Question from the PDF Files", key="user_question")

    with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader("Upload your PDF Files and Click on the Submit & Process Button", accept_multiple_files=True)
        if st.button("Submit & Process"):
            with st.spinner("Processing..."):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                get_vector_store(text_chunks)
                st.success("Done")

    if user_question:
        response = user_input(user_question, pdf_docs.copy())  # Pass a copy of pdf_docs to avoid modification
        if isinstance(response, dict) and "error" in response:
            st.error(response["error"])  # Display error message
        else:
            # Append user question and response to chat history
            answer_text = response["output_text"]
            if "answer_location" in response:
                answer_location = response["answer_location"]
            else:
                answer_location = ""
            st.session_state["chat_history"].append({"question": user_question, "answer": answer_text, "location": answer_location})

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
                answer_location = entry["location"]
                st.write(
                    f'<div style="background-color: lightblue; padding: 10px; margin: 5px;"><b>Answer:</b> {answer_text} </div> <div style="color: grey; padding: 10px; margin: 5px;">{answer_location}</div>',
                    unsafe_allow_html=True,
                )

if __name__ == "__main__":
    main()
