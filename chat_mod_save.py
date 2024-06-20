import os



from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.vectorstores import FAISS
from dotenv import load_dotenv


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



 



if __name__ == "__main__":



    # Replace 'your_pdf_files' with the actual PDF files



    your_pdf_files = ["manual1.pdf"]  # Add your PDF file paths here



    process_pdfs(your_pdf_files)