

import os  
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader  
from langchain.text_splitter import RecursiveCharacterTextSplitter  
from langchain_huggingface import HuggingFaceEmbeddings  
from langchain_community.vectorstores import FAISS  

# Step 1: Load raw PDF(s)
DATA_PATH = "data/"  

def load_pdf_files(data: str):
    if not os.path.isdir(data):
        raise FileNotFoundError(f"DATA_PATH not found: {data}")  
    loader = DirectoryLoader(
        data,
        glob="**/*.pdf",  
        loader_cls=PyPDFLoader
    )
    documents = loader.load()
    return documents 

documents = load_pdf_files(DATA_PATH)
print("Loaded PDF pages:", len(documents)) 

# Step 2: Create Chunks
def create_chunks(extracted_data):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    text_chunks = text_splitter.split_documents(extracted_data)
    return text_chunks 

text_chunks = create_chunks(documents)
print(" Created text chunks:", len(text_chunks))  

# Step 3: Create Embeddings (no API key needed)
def get_embedding_model():
    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    return embedding_model  # [web:22]

embedding_model = get_embedding_model()

# Step 4: Store embeddings in FAISS
DB_FAISS_PATH = "vectorstore/db_faiss"  
os.makedirs(DB_FAISS_PATH, exist_ok=True)
db = FAISS.from_documents(text_chunks, embedding_model)
db.save_local(DB_FAISS_PATH)
print("FAISS vectorstore saved at:", DB_FAISS_PATH)  
