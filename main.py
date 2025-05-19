from fastapi import FastAPI, UploadFile, File, Form, Request
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
import tempfile
import os
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

# Global FAISS index and retriever
faiss_index = None
retriever = None
qa_chain = None

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

    # Load PDF and split into documents
    loader = PyMuPDFLoader(tmp_path)
    docs = loader.load()

    # Extract texts and metadata from loaded documents
    texts = [doc.page_content for doc in docs]

    # Build FAISS vector store from embeddings
    faiss_index = FAISS.from_texts(texts, embeddings)

    # Setup retriever from FAISS index
    retriever = faiss_index.as_retriever()

    # Create RetrievalQA chain with LLM and retriever
    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

    # Remove temporary file
    os.unlink(tmp_path)

    return {"message": f"Uploaded and indexed {file.filename}", "num_docs": len(texts)}

@app.post("/ask/")
async def ask_question(question: str = Form(...)):
    global qa_chain
    if not qa_chain:
        return {"error": "No documents indexed yet. Please upload a PDF first."}

    answer = qa_chain.run(question)
    return {"response": answer}
