# build_vector_store.py - FIXED VERSION

from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.api.types import EmbeddingFunction, Documents, Embeddings
from ingest import load_and_chunk

# Define the embedding class
class LocalHuggingFaceEmbedding(EmbeddingFunction[Documents]):
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

    def __call__(self, docs: Documents) -> Embeddings:
        embeddings = self.model.encode(docs)
        return embeddings.tolist()# build_vector_store.py - COMPLETE FIXED VERSION

from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.api.types import EmbeddingFunction, Documents, Embeddings
from ingest import load_and_chunk

class LocalHuggingFaceEmbedding(EmbeddingFunction[Documents]):
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

    def __call__(self, docs: Documents) -> Embeddings:
        embeddings = self.model.encode(docs)
        # CRITICAL: Convert numpy array to Python list
        return embeddings.tolist() 

def build_index(pdf_path="contract.pdf"):
    print("📚 Loading and chunking PDF...")
    chunks = load_and_chunk(pdf_path)
    print(f"✅ Got {len(chunks)} chunks.")
    
    print("🧠 Initializing ChromaDB...")
    client = chromadb.PersistentClient(path="./chroma_store")
    
    try:
        client.delete_collection(name="loan_contract_chunks")
        print("🗑️  Deleted old collection")
    except:
        pass
    
    ef = LocalHuggingFaceEmbedding()
    collection = client.create_collection(
        name="loan_contract_chunks",
        embedding_function=ef
    )
    print("✅ Created fresh collection")
    
    print("📥 Adding chunks...")
    documents = [chunk.page_content for chunk in chunks]
    metadatas = [chunk.metadata for chunk in chunks]
    ids = [f"chunk-{i}" for i in range(len(chunks))]
    
    collection.add(
        ids=ids,
        documents=documents,
        metadatas=metadatas
    )
    
    print(f"✅ Indexed {len(documents)} chunks into Chroma.")
    print("🚀 Vector store ready!")

if __name__ == "__main__":
    build_index()


def build_index(pdf_path="contract.pdf"):
    print("📚 Loading and chunking PDF...")
    chunks = load_and_chunk(pdf_path)
    print(f"✅ Got {len(chunks)} chunks.")
    
    print("🧠 Initializing ChromaDB...")
    client = chromadb.PersistentClient(path="./chroma_store")
    
    # Delete old collection if it exists
    try:
        client.delete_collection(name="loan_contract_chunks")
        print("🗑️  Deleted old collection")
    except:
        print("ℹ️  No old collection to delete")
    
    # Create fresh collection with embedding function
    ef = LocalHuggingFaceEmbedding()
    collection = client.create_collection(
        name="loan_contract_chunks",
        embedding_function=ef
    )
    print("✅ Created fresh collection")
    
    print("📥 Adding chunks...")
    documents = [chunk.page_content for chunk in chunks]
    metadatas = [chunk.metadata for chunk in chunks]
    ids = [f"chunk-{i}" for i in range(len(chunks))]
    
    # Add to collection
    collection.add(
        ids=ids,
        documents=documents,
        metadatas=metadatas
    )
    
    print(f"✅ Indexed {len(documents)} chunks into Chroma.")
    print("🚀 Vector store ready!")

if __name__ == "__main__":
    build_index()
