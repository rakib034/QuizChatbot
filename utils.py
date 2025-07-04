# utils.py

import fitz  # PyMuPDF
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
import os


def load_pdfs_from_folder_with_page_metadata(folder_path):
    """
    Loads all PDFs in a folder, extracts text page by page,
    and attaches page number and source file name as metadata.
    """
    documents = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".pdf"):
            pdf_path = os.path.join(folder_path, filename)
            with fitz.open(pdf_path) as doc:
                for page_num in range(len(doc)):
                    page = doc.load_page(page_num)
                    text = page.get_text()
                    documents.append(Document(
                        page_content=text,
                        metadata={
                            "page": page_num + 1,
                            "source": filename
                        }
                    ))
    return documents


def chunk_documents_with_metadata(documents, chunk_size=500, chunk_overlap=50):
    """
    Splits documents into chunks and keeps metadata (page number, source).
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    all_chunks = []
    for doc in documents:
        chunks = splitter.split_text(doc.page_content)
        for chunk_text in chunks:
            all_chunks.append(Document(
                page_content=chunk_text,
                metadata=doc.metadata
            ))
    return all_chunks


def get_embedder():
    """
    Returns a sentence-transformer embedder (MiniLM).
    """
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


def create_faiss_index(documents, embeddings, save_path="faiss_index"):
    """
    Creates and saves a FAISS index from embedded documents.
    """
    db = FAISS.from_documents(documents, embeddings)
    db.save_local(save_path)
    return db


def load_faiss_index(embeddings, path="faiss_index"):
    """
    Loads a FAISS index from local storage.
    """
    return FAISS.load_local(path, embeddings, allow_dangerous_deserialization=True)
