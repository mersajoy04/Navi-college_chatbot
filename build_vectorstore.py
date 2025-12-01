from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    PyPDFLoader, UnstructuredWordDocumentLoader, TextLoader, UnstructuredHTMLLoader
)
from pathlib import Path

# Load documents from college_documents and website
def load_docs():
    docs = []

    # Load from college_documents
    for file in Path("college_documents").rglob("*"):
        if file.suffix == ".pdf":
            docs.extend(PyPDFLoader(str(file)).load())
        elif file.suffix == ".docx":
            docs.extend(UnstructuredWordDocumentLoader(str(file)).load())
        elif file.suffix == ".txt":
            docs.extend(TextLoader(str(file)).load())

    # Load from website clone
    for file in Path("mainproject/www.stvincentngp.edu.in").rglob("*.html"):
        docs.extend(UnstructuredHTMLLoader(str(file)).load())

    return docs

# Main
if __name__ == "__main__":
    print("📂 Loading all documents...")
    raw_docs = load_docs()
    print(f"✅ Loaded {len(raw_docs)} files.")

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    chunks = splitter.split_documents(raw_docs)

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    vectorstore = FAISS.from_documents(chunks, embeddings)

    vectorstore.save_local("college_vector_db")
    print("✅ Saved FAISS vector DB: college_vector_db")
