import os
import tempfile
import traceback

import ebooklib
import mobi
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File, Form, Request
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from ebooklib import epub

from langchain.chains import RetrievalQA
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_mistralai import ChatMistralAI, MistralAIEmbeddings


# --------------------------------------------------
# Configuration
# --------------------------------------------------

load_dotenv()

MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")

if not MISTRAL_API_KEY:
    raise RuntimeError("MISTRAL_API_KEY is not set in your .env file")


app = FastAPI()
templates = Jinja2Templates(directory="templates")
app.mount(
    "/assets",
    StaticFiles(directory="assets"),
    name="assets"
)

# --------------------------------------------------
# Mistral
# --------------------------------------------------

llm = ChatMistralAI(
    model="mistral-small-latest",
    api_key=MISTRAL_API_KEY,
)

embeddings = MistralAIEmbeddings(
    model="mistral-embed",
    api_key=MISTRAL_API_KEY,
)


# --------------------------------------------------
# Global RAG state
# --------------------------------------------------

faiss_index = None
retriever = None
qa_chain = None


# --------------------------------------------------
# EPUB extraction
# --------------------------------------------------

def extract_epub_text(epub_path: str) -> str:
    try:
        book = epub.read_epub(epub_path)
        texts = []

        items = list(book.get_items())

        print(f"EPUB content items found: {len(items)}")

        for item in items:
            if item.get_type() != ebooklib.ITEM_DOCUMENT:
                continue

            try:
                content = item.get_content()

                soup = BeautifulSoup(
                    content,
                    features="html.parser"
                )

                extracted_text = soup.get_text(
                    separator="\n",
                    strip=True
                )

                print(
                    f"Extracted text from item: "
                    f"{len(extracted_text)} characters"
                )

                if extracted_text:
                    texts.append(extracted_text)

            except Exception as e:
                print(f"Error extracting EPUB item: {e}")

        result = "\n\n".join(texts)

        print(
            f"Total EPUB text extracted: "
            f"{len(result)} characters"
        )

        return result

    except Exception as e:
        print(f"Error in extract_epub_text: {e}")
        raise


# --------------------------------------------------
# MOBI extraction
# --------------------------------------------------

def extract_mobi_text(mobi_path: str) -> str:
    """
    Extract text from MOBI/AZW/AZW3 using mobi 0.3.3.

    mobi.extract() unpacks the ebook and returns:
        (temporary_directory, extracted_file_path)
    """

    try:
        print(f"Extracting Kindle file: {mobi_path}")

        tempdir, extracted_path = mobi.extract(mobi_path)

        print(f"MOBI extraction directory: {tempdir}")
        print(f"Extracted file: {extracted_path}")

        if not os.path.exists(extracted_path):
            raise ValueError(
                f"MOBI extraction failed: "
                f"{extracted_path} does not exist"
            )

        # The mobi package normally extracts the book into
        # an HTML-like file. Read it and strip the markup.
        with open(
            extracted_path,
            "rb"
        ) as f:
            raw_content = f.read()

        # Try UTF-8 first, then fall back gracefully.
        content = raw_content.decode(
            "utf-8",
            errors="ignore"
        )

        soup = BeautifulSoup(
            content,
            features="html.parser"
        )

        text = soup.get_text(
            separator="\n",
            strip=True
        )

        print(
            f"Total MOBI text extracted: "
            f"{len(text)} characters"
        )

        return text

    except Exception as e:
        print(f"Error in extract_mobi_text: {e}")
        raise


# --------------------------------------------------
# Create documents from text
# --------------------------------------------------

def text_to_documents(
    text: str,
    filename: str
):
    if not text or not text.strip():
        raise ValueError(
            "No text could be extracted from the document."
        )

    text_length = len(text)

    docs = []

    # Break very large extracted documents into sections
    if text_length > 30000:

        print(
            f"Large document detected: "
            f"{text_length} characters"
        )

        section_size = 20000

        for i in range(
            0,
            text_length,
            section_size
        ):
            section = text[
                i:i + section_size
            ]

            if section.strip():
                docs.append(
                    Document(
                        page_content=section,
                        metadata={
                            "source": filename,
                            "section": i // section_size,
                        },
                    )
                )

    else:

        docs.append(
            Document(
                page_content=text,
                metadata={
                    "source": filename
                },
            )
        )

    return docs


# --------------------------------------------------
# Home
# --------------------------------------------------

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):

    return templates.TemplateResponse(
        "index.html",
        {
            "request": request
        }
    )


# --------------------------------------------------
# Upload document
# --------------------------------------------------

@app.post("/upload/")
async def upload_file(
    file: UploadFile = File(...)
):

    global faiss_index
    global retriever
    global qa_chain

    filename = (
        file.filename or "uploaded_file"
    ).lower()

    suffix = os.path.splitext(
        filename
    )[1]

    supported = {
        ".pdf",
        ".epub",
        ".mobi",
        ".azw",
        ".azw3",
    }

    if suffix not in supported:
        return {
            "error": (
                f"Unsupported file type: {suffix}. "
                "Supported: PDF, EPUB, MOBI, AZW, AZW3."
            )
        }

    tmp_path = None

    try:

        # ------------------------------------------
        # Save uploaded file
        # ------------------------------------------

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=suffix
        ) as tmp:

            content = await file.read()

            tmp.write(content)

            tmp_path = tmp.name

        print(
            f"Processing file: "
            f"{filename} ({suffix})"
        )

        docs = []

        # ------------------------------------------
        # PDF
        # ------------------------------------------

        if suffix == ".pdf":

            loader = PyMuPDFLoader(
                tmp_path
            )

            docs = loader.load()

            print(
                f"PDF loaded: "
                f"{len(docs)} pages"
            )

        # ------------------------------------------
        # EPUB
        # ------------------------------------------

        elif suffix == ".epub":

            text = extract_epub_text(
                tmp_path
            )

            docs = text_to_documents(
                text,
                filename
            )

            print(
                f"Created {len(docs)} EPUB documents"
            )

        # ------------------------------------------
        # MOBI / AZW / AZW3
        # ------------------------------------------

        elif suffix in {
            ".mobi",
            ".azw",
            ".azw3",
        }:

            text = extract_mobi_text(
                tmp_path
            )

            docs = text_to_documents(
                text,
                filename
            )

            print(
                f"Created {len(docs)} "
                f"Kindle documents"
            )

        # ------------------------------------------
        # Validate documents
        # ------------------------------------------

        if not docs:

            raise ValueError(
                "No valid documents were created."
            )

        for i, doc in enumerate(docs[:2]):

            print(
                f"Document {i}: "
                f"{len(doc.page_content)} characters"
            )

            print(
                f"Sample: "
                f"{doc.page_content[:100].strip()}..."
            )

        # ------------------------------------------
        # Chunk documents
        # ------------------------------------------

        text_splitter = (
            RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200,
            )
        )

        split_docs = (
            text_splitter.split_documents(
                docs
            )
        )

        print(
            f"Split into "
            f"{len(split_docs)} chunks"
        )

        # ------------------------------------------
        # If too few chunks, split smaller
        # ------------------------------------------

        if len(split_docs) < 3:

            print(
                "Very few chunks created. "
                "Trying smaller chunks."
            )

            text_splitter = (
                RecursiveCharacterTextSplitter(
                    chunk_size=500,
                    chunk_overlap=100,
                )
            )

            split_docs = (
                text_splitter.split_documents(
                    docs
                )
            )

            print(
                f"Re-split into "
                f"{len(split_docs)} chunks"
            )

        # ------------------------------------------
        # Create FAISS vector index
        # ------------------------------------------

        print(
            "Creating FAISS index "
            "with Mistral embeddings..."
        )

        faiss_index = FAISS.from_documents(
            split_docs,
            embeddings
        )

        # ------------------------------------------
        # Configure retriever
        # ------------------------------------------

        retriever = faiss_index.as_retriever(
            search_type="mmr",
            search_kwargs={
                "k": 6,
                "fetch_k": 10,
            },
        )

        # ------------------------------------------
        # Create RAG chain
        # ------------------------------------------

        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=retriever,
            chain_type="stuff",
            return_source_documents=True,
            verbose=True,
        )

        return {
            "message": (
                f"Uploaded and indexed "
                f"{file.filename} "
                f"({len(split_docs)} chunks)"
            ),
            "chunks": len(split_docs),
        }

    except Exception as e:

        print(
            f"Error processing file: {e}"
        )

        print(
            traceback.format_exc()
        )

        return {
            "error": (
                f"Failed to process file: {e}"
            )
        }

    finally:

        if tmp_path and os.path.exists(
            tmp_path
        ):

            try:
                os.unlink(tmp_path)

            except Exception:
                pass


# --------------------------------------------------
# Ask question
# --------------------------------------------------

@app.post(
    "/ask/",
    response_class=HTMLResponse
)
async def ask_question(
    question: str = Form(...)
):

    global qa_chain

    if not qa_chain:

        return HTMLResponse(
            "No documents indexed yet. "
            "Please upload a file first.",
            status_code=400,
        )

    try:

        print(
            f"Processing question: "
            f"{question}"
        )

        response = qa_chain.invoke(
            {
                "query": question
            }
        )

        print(
            f"Response keys: "
            f"{response.keys()}"
        )

        result = response.get(
            "result",
            ""
        )

        if not result:

            return HTMLResponse(
                "The model did not return an answer.",
                status_code=500,
            )

        return HTMLResponse(
            content=result
        )

    except Exception as e:

        print(
            f"Error in /ask/: {e}"
        )

        print(
            traceback.format_exc()
        )

        return HTMLResponse(
            "Sorry, something went wrong.",
            status_code=500,
        )


# --------------------------------------------------
# Clear session
# --------------------------------------------------

@app.post("/clear/")
async def clear_session():

    global faiss_index
    global retriever
    global qa_chain

    faiss_index = None
    retriever = None
    qa_chain = None

    return JSONResponse(
        content={
            "status": "cleared"
        }
    )