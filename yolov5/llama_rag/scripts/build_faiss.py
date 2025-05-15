from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.docstore.document import Document
import os

# Load your knowledge base text
with open("../data/rag_docs.txt", "r") as f:
    text = f.read()

# Split into chunks
chunks = text.split("\n\n")

# Wrap each chunk into a Document
docs = [Document(page_content=chunk) for chunk in chunks if chunk.strip()]

# Use sentence-transformers for embeddings
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Create FAISS index
db = FAISS.from_documents(docs, embeddings)

# Save it
db.save_local("rag_index")
print("✅ FAISS index saved in rag_index/")
