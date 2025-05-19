from fastapi import FastAPI, UploadFile, File, Form, Request
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, JSONResponse
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
import tempfile
import os
import uuid
from dotenv import load_dotenv

app = FastAPI()
templates = Jinja2Templates(directory="templates")

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

llm = ChatOpenAI(
    model_name="gpt-4o-mini",
    openai_api_key=OPENAI_API_KEY
)

embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

faiss_index = None
retriever = None
qa_chain = None

# In-memory store of session_id -> conversation history
conversations = {}

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/upload/")
async def upload_pdf(file: UploadFile = File(...)):
    global faiss_index, retriever, qa_chain

    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    # Load using PyMuPDFLoader
    loader = PyMuPDFLoader(tmp_path)
    docs = loader.load()

    # Split into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,   # adjust for your token limit
        chunk_overlap=150
    )
    split_docs = text_splitter.split_documents(docs)

    # Create FAISS index from split docs
    faiss_index = FAISS.from_documents(split_docs, embeddings)
    retriever = faiss_index.as_retriever()
    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

    os.unlink(tmp_path)

    return {
        "message": f"✅ Uploaded and indexed {file.filename}",
        "chunks": len(split_docs)
    }

@app.post("/ask/", response_class=HTMLResponse)
async def ask_question(
    question: str = Form(...)
):
    global qa_chain

    if not qa_chain:
        return HTMLResponse("No documents indexed yet. Please upload a file first.", status_code=400)

    try:
        answer = qa_chain.run(question)
        return HTMLResponse(content=answer)

    except Exception as e:
        print("Error in /ask/:", e)
        return HTMLResponse("Sorry, something went wrong. Please try again.", status_code=500)

