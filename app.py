import ebooklib
from fastapi import FastAPI, UploadFile, File, Form, Request
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, JSONResponse
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

import tempfile
import os
import uuid
from dotenv import load_dotenv

# EPUB parsing
from ebooklib import epub
from bs4 import BeautifulSoup

# MOBI handling
from mobi import Mobi

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


def extract_epub_text(epub_path: str) -> str:
    try:
        book = epub.read_epub(epub_path)
        texts = []
        
        print("EPUB content items found:", len(list(book.get_items())))
        
        # Extract text from all items that could contain content
        for item in book.get_items():
            if item.get_type() == ebooklib.ITEM_DOCUMENT:
                try:
                    content = item.get_content()
                    soup = BeautifulSoup(content, features="html.parser")
                    extracted_text = soup.get_text()
                    print(f"Extracted text from item: {len(extracted_text)} characters")
                    if len(extracted_text.strip()) > 0:
                        texts.append(extracted_text)
                except Exception as e:
                    print(f"Error extracting from EPUB item: {str(e)}")
        
        result = "\n\n".join(texts)
        print(f"Total EPUB text extracted: {len(result)} characters")
        return result
    except Exception as e:
        print(f"Error in extract_epub_text: {str(e)}")
        raise


def extract_mobi_text(mobi_path: str) -> str:
    try:
        mobi_file = Mobi(mobi_path)
        mobi_file.parse()
        
        # Try multiple methods to extract content
        all_text = []
        
        # Method 1: Try text_section from book_header
        try:
            if hasattr(mobi_file, 'book_header') and hasattr(mobi_file.book_header, 'text_section'):
                text = mobi_file.book_header.text_section
                if text and len(text) > 0:
                    print(f"Method 1: Got {len(text)} characters")
                    all_text.append(text)
        except Exception as e:
            print(f"Method 1 error: {str(e)}")
        
        # Method 2: Try get_text method
        try:
            if hasattr(mobi_file, 'get_text'):
                text = mobi_file.get_text()
                if text and len(text) > 0:
                    print(f"Method 2: Got {len(text)} characters")
                    all_text.append(text)
        except Exception as e:
            print(f"Method 2 error: {str(e)}")
        
        # Method 3: Try raw content extraction
        try:
            if hasattr(mobi_file, 'records'):
                for record in mobi_file.records:
                    if isinstance(record, bytes) or isinstance(record, str):
                        try:
                            if isinstance(record, bytes):
                                text = record.decode('utf-8', errors='ignore')
                            else:
                                text = record
                            if text and len(text) > 20:  # Reasonable text size
                                print(f"Method 3: Got {len(text)} characters from record")
                                all_text.append(text)
                        except Exception as e:
                            pass  # Ignore decoding errors
        except Exception as e:
            print(f"Method 3 error: {str(e)}")
            
        # Combine all text
        result = "\n\n".join(all_text)
        print(f"Total MOBI text extracted: {len(result)} characters")
        return result
        
    except Exception as e:
        print(f"Error in extract_mobi_text: {str(e)}")
        raise


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/upload/")
async def upload_file(file: UploadFile = File(...)):
    global faiss_index, retriever, qa_chain

    filename = file.filename.lower()
    suffix = os.path.splitext(filename)[1]

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        print(f"Processing file: {filename} (type: {suffix})")
        docs = []
        
        if suffix == ".pdf":
            # Use PyMuPDFLoader for PDF
            loader = PyMuPDFLoader(tmp_path)
            docs = loader.load()
            print(f"PDF loaded: {len(docs)} pages")

        elif suffix == ".epub":
            # Extract text from EPUB manually
            text = extract_epub_text(tmp_path)
            text_length = len(text)
            print(f"EPUB extracted text length: {text_length}")
            
            if not text.strip():
                raise ValueError("No text extracted from EPUB.")
            
            # Split the long text into multiple documents to avoid context issues
            # Create smaller chunks first
            if text_length > 30000:
                print("EPUB text is very long, splitting into initial chunks")
                # First split into large sections
                section_size = 20000
                sections = []
                for i in range(0, text_length, section_size):
                    section = text[i:i+section_size]
                    if section.strip():
                        sections.append(section)
                
                print(f"Created {len(sections)} initial sections")
                # Create document for each large section
                for i, section in enumerate(sections):
                    docs.append(Document(
                        page_content=section, 
                        metadata={"source": filename, "section": i}
                    ))
            else:
                # Create a single document if the text is manageable
                docs = [Document(page_content=text, metadata={"source": filename})]
                
            print(f"Created {len(docs)} EPUB documents")

        elif suffix in [".mobi", ".azw", ".azw3"]:
            # Use mobi lib for Kindle formats
            text = extract_mobi_text(tmp_path)
            text_length = len(text)
            print(f"MOBI extracted text length: {text_length}")
            
            if not text:
                raise ValueError("Could not extract text from Kindle file.")
            
            # Similar approach to EPUB - split large texts
            if text_length > 30000:
                print("MOBI text is very long, splitting into initial chunks")
                section_size = 20000
                sections = []
                for i in range(0, text_length, section_size):
                    section = text[i:i+section_size]
                    if section.strip():
                        sections.append(section)
                
                print(f"Created {len(sections)} initial sections")
                # Create document for each large section
                for i, section in enumerate(sections):
                    docs.append(Document(
                        page_content=section, 
                        metadata={"source": filename, "section": i}
                    ))
            else:
                # Create a single document if the text is manageable
                docs = [Document(page_content=text, metadata={"source": filename})]
                
            print(f"Created {len(docs)} MOBI documents")

        else:
            os.unlink(tmp_path)
            return {"error": f"Unsupported file type: {suffix}. Supported: PDF, EPUB, MOBI, AZW."}

        # Check if we have valid documents
        if not docs or len(docs) == 0:
            raise ValueError("No valid documents were created from the file.")
            
        # Check content of first document
        if not docs[0].page_content or len(docs[0].page_content.strip()) < 50:
            raise ValueError("Document content appears to be empty or too short.")

        # Print document content for debugging
        for i, doc in enumerate(docs[:2]):  # Show first couple docs
            print(f"Document {i} content sample: {doc.page_content[:100].strip()}...")
            print(f"Document {i} length: {len(doc.page_content)} chars")

        # Split into chunks with adjusted parameters for better retrieval
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,  # Smaller chunks for more precise retrieval
            chunk_overlap=200  # More overlap to preserve context
        )
        
        split_docs = text_splitter.split_documents(docs)
        print(f"Split into {len(split_docs)} chunks")
        
        # Check if we have enough chunks
        if len(split_docs) < 3:
            print("WARNING: Very few chunks created. Text may not have been properly extracted.")
            # Try with smaller chunk size for low content
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=500,  
                chunk_overlap=100
            )
            split_docs = text_splitter.split_documents(docs)
            print(f"Re-split into {len(split_docs)} smaller chunks")

        # Create FAISS index
        faiss_index = FAISS.from_documents(split_docs, embeddings)
        
        # Configure retriever for better results
        retriever = faiss_index.as_retriever(
            search_type="mmr",  # Maximum marginal relevance for diversity
            search_kwargs={
                "k": 6,  # Retrieve more documents
                "fetch_k": 10  # Consider more documents for diversity
            }
        )
        
        # Create QA chain with specific parameters for better answers
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm, 
            retriever=retriever,
            chain_type="stuff",
            return_source_documents=True,
            verbose=True
        )

        os.unlink(tmp_path)

        return {
            "message": f"✅ Uploaded and indexed {file.filename} ({len(split_docs)} chunks)",
            "chunks": len(split_docs)
        }

    except Exception as e:
        import traceback
        print(f"Error processing file: {str(e)}")
        print(traceback.format_exc())
        try:
            os.unlink(tmp_path)
        except:
            pass
        return {"error": f"Failed to process file: {str(e)}"}


@app.post("/ask/", response_class=HTMLResponse)
async def ask_question(question: str = Form(...)):
    global qa_chain

    if not qa_chain:
        return HTMLResponse("No documents indexed yet. Please upload a file first.", status_code=400)

    try:
        print(f"Processing question: {question}")
        
        # First attempt with more verbose debugging
        try:
            # Run direct query with detailed parameters
            response = qa_chain({
                "query": question,
                "return_only_outputs": False
            })
            
            # Log the full response structure
            print(f"Response keys: {response.keys() if isinstance(response, dict) else type(response)}")
            
            if isinstance(response, dict) and "result" in response:
                result = response["result"]
                print(f"Answer found in result key, length: {len(result)}")
                
                # Check if we got an "I don't know" type response
                if "don't know" in result.lower() or "unknown" in result.lower() or "no information" in result.lower():
                    print("Got 'don't know' response, trying direct QA...")
                    # Try direct LLM call with retrieved context
                    if "source_documents" in response and response["source_documents"]:
                        retrieved_docs = response["source_documents"]
                        print(f"Found {len(retrieved_docs)} source documents")
                        
                        # Extract text from retrieved documents
                        contexts = [doc.page_content for doc in retrieved_docs]
                        combined_context = "\n\n".join(contexts)
                        print(f"Combined context length: {len(combined_context)}")
                        
                        # Direct prompt using retrieved information
                        if len(combined_context) > 100:  # Ensure we have meaningful context
                            direct_prompt = f"""Based on the following context from the document:
                            
{combined_context[:9000]}

Question: {question}

Answer the question based only on the provided context. If the context doesn't contain the answer, say so."""
                            
                            direct_result = llm.invoke(direct_prompt)
                            print(f"Direct LLM result: {direct_result.content[:100]}...")
                            return HTMLResponse(content=direct_result.content)
                
                return HTMLResponse(content=result)
            else:
                print("Result key not found in response")
        except Exception as e:
            print(f"First attempt error: {str(e)}")
            import traceback
            print(traceback.format_exc())
        
        # Fallback to direct run method
        print("Falling back to direct qa_chain.run method")
        result = qa_chain.run(question)
        print(f"Fallback answer length: {len(result)}")
        return HTMLResponse(content=result)

    except Exception as e:
        import traceback
        print(f"Error in /ask/: {str(e)}")
        print(traceback.format_exc())
        return HTMLResponse("Sorry, something went wrong. Please try again.", status_code=500)


@app.post("/clear/")
async def clear_session():
    global faiss_index, retriever, qa_chain
    faiss_index = None
    retriever = None
    qa_chain = None
    return JSONResponse(content={"status": "cleared"})