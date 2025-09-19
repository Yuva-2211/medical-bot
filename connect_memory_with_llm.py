
import os
from dotenv import load_dotenv

from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace, HuggingFaceEmbeddings  
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_core.prompts import ChatPromptTemplate  

# 1) Load env
load_dotenv()  
HF_TOKEN = os.getenv("HF_TOKEN")

# 2) Model repo (providers often expose as conversational)
HUGGINGFACE_REPO_ID = "mistralai/Mistral-7B-Instruct-v0.2" 
# 3) Build Endpoint LLM and wrap as Chat
def load_chat_model(repo_id: str):
    if not HF_TOKEN:
        raise EnvironmentError("HF_TOKEN not found in environment/.env")  
    endpoint_llm = HuggingFaceEndpoint(
        repo_id=repo_id,
        huggingfacehub_api_token=HF_TOKEN,
        task="conversational",    
        temperature=0.5,
        max_new_tokens=512,
       
    )
   
    chat_model = ChatHuggingFace(llm=endpoint_llm)
    return chat_model

# 4) Embeddings
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)  

# 5) Load FAISS
DB_FAISS_PATH = "vectorstore/db_faiss"
if os.path.isdir(DB_FAISS_PATH) and any(
    f.endswith(".faiss") or f.endswith(".pkl") for f in os.listdir(DB_FAISS_PATH)
):
    vectorstore = FAISS.load_local(
        DB_FAISS_PATH,
        embeddings=embedding_model,
        allow_dangerous_deserialization=True  
    )
else:
  
    texts = ["Cancer is a disease.", "Diabetes affects insulin production."]
    vectorstore = FAISS.from_texts(texts, embedding=embedding_model)

# 6) Chat-style prompt (messages)
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful domain assistant. Use the retrieved context to answer."),  
        ("human", "Context:\n{context}\n\nQuestion: {question}\nAnswer:")  
    ]
)

# 7) RetrievalQA chain with chat model
chat_model = load_chat_model(HUGGINGFACE_REPO_ID)
qa_chain = RetrievalQA.from_chain_type(
    llm=chat_model,                           
    retriever=vectorstore.as_retriever(),
    chain_type_kwargs={"prompt": prompt},
    return_source_documents=False
)


if __name__ == "__main__":
    print("Type 'exit' or 'quit' to stop.")  
    while True:
        user_query = input("Write Query Here: ")
        if user_query.lower() in ["exit", "quit"]:
            break
        response = qa_chain.invoke({"query": user_query})  
        print("Answer:", response["result"])
