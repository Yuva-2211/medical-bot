

# 🏥 Medical Bot – PDF-powered QA Assistant

This project is a medical question-answering bot built using LangChain, Hugging Face, FAISS, and Streamlit.
It lets you upload medical knowledge (e.g., textbooks, encyclopedias) as PDFs, splits them into chunks, creates embeddings, stores them in a FAISS vector database, and then allows you to query them using a Hugging Face LLM (Mistral-7B).

## 📂 Project Structure
medical-bot/
│── data/                  # Place your PDFs here (not uploaded in repo)
│── vectorstore/           # Generated FAISS database (auto-created after running script)
│── app.py                 # Streamlit app (if you want UI)
│── build_vectorstore.py   # Script to process PDFs and build FAISS DB
│── chat_qa.py             # CLI chatbot script
│── requirements.txt       # Dependencies
│── README.md              # Project documentation

## ⚙️ Setup
1. Clone the Repository
git clone https://github.com/Yuva-2211/medical-bot.git
cd medical-bot

2. Create Virtual Environment
python -m venv venv
source venv/bin/activate    # On Mac/Linux
venv\Scripts\activate       # On Windows

3. Install Dependencies
pip install -r requirements.txt

4. Add Your PDFs

Place all your medical PDFs inside the data/ folder.

Example: data/Gale_Encyclopedia_Medicine_V3.pdf

5. Hugging Face Token

Create a .env file in the root directory:

HF_TOKEN=your_huggingface_api_token

🏗️ Build Vector Store

Run the following command to load PDFs, split them into chunks, embed them, and save them to FAISS:

python build_vectorstore.py


✅ This will create vectorstore/db_faiss/ with the FAISS index.

🤖 Run the Chatbot
CLI Mode
python chat_qa.py


Type your question in the terminal.

Type exit or quit to stop.



## 📊 How It Works

Load PDFs → Extract text from data/

Chunking → Split into ~500-character overlapping chunks

Embeddings → Convert chunks into vector embeddings with sentence-transformers/all-MiniLM-L6-v2

Store in FAISS → Save embeddings for fast retrieval

Query → Retrieve relevant chunks and feed them to the Hugging Face model (Mistral-7B-Instruct-v0.2) for responses

## 🚀 Future Improvements

Add support for more document formats (DOCX, TXT)

Improve UI with chat history

Deploy on Hugging Face Spaces or Streamlit Cloud

## 🔗 Links

GitHub Repo: [https://github.com/Yuva-2211/medical-bot.git]

Hugging Face Model: Mistral-7B-Instruct-v0.2

---

## 📜 License  

This project is licensed under the **MIT License** – see the [LICENSE](./LICENSE) file for details.



