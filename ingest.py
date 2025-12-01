from langchain.document_loaders import DirectoryLoader, PyPDFLoader, TextLoader, Docx2txtLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os

# Load all documents from the folder
loader = DirectoryLoader("college_documents", glob="**/*.*")
documents = loader.load()

# Split the documents into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
texts = text_splitter.split_documents(documents)

# ---------- UNIT TEST: Chunk Size Validation ----------
print("🔍 Running chunk size validation...")
print("Number of chunks:", len(texts))

try:
    for c in texts:
        assert len(c.page_content) < 600
    print("✅ Unit Test Passed: Chunk sizes valid")
except AssertionError:
    print("❌ Unit Test Failed: A chunk exceeds 600 characters")

# Create embeddings
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

# Create FAISS vector DB
vectorstore = FAISS.from_documents(texts, embeddings)

# Save vector DB locally
vectorstore.save_local("college_vector_db")

print("✅ Vector database created and saved.")
