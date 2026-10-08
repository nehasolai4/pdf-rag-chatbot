from fastapi import FastAPI,UploadFile, File
from pypdf import PdfReader
import os

from langchain_text_splitters import RecursiveCharacterTextSplitter
from embeddings import create_embeddings
from vector_store import create_vector_store, search
from llm import generate_answer


app=FastAPI()

vector_store=None
chunks_store=[]

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@app.get("/")
def root():
    return {"message":"PDF RAG Chatbot API is running"}


def create_chunks(text):
    splitter=RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=["\n\n", "\n", " ", ""]
    )
    return splitter.split_text(text)


@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    global vector_store, chunks_store
    file_path = os.path.join(UPLOAD_DIR, file.filename)

    with open(file_path, "wb") as buffer:
        buffer.write(await file.read())

    reader = PdfReader(file_path)
    text = ""

    for page in reader.pages:
        page_text = page.extract_text()

        if page_text:
            text += page_text+"\n"

    chunks = create_chunks(text)

    embeddings = create_embeddings(chunks)

    index = create_vector_store(embeddings)

    vector_store = index
    chunks_store = chunks

    # return {
    #     "filename": file.filename,
    #     "pages": len(reader.pages),
    #     "number_of_chunks": len(chunks),
    #     "chunks": chunks
    # }

    return {
        "filename": file.filename,
        "pages": len(reader.pages),
        "number_of_chunks": len(chunks),
        "embedding_dimensions": len(embeddings[0])
    }


@app.get("/ask")
def ask_question(question:str):

    
    # 1. Convert question into an embedding
    if vector_store is None:
        return {"error":"Please upload a PDF first"}

    query_embedding = create_embeddings([question])[0]

    # 2. Retrieve relevant chunks
    results=search(
        vector_store,
        query_embedding,
        chunks_store,
        k=3
    )

    # 3. Combine retrieved chunks into context
    context="\n\n".join(
        result["chunk"] for result in results
    )

    # 4. Give question + retrived context to Gemini
    answer = generate_answer(question,context)

    return{
        "question":question,
        "answer":answer,
        "sources":results
    }