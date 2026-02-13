import chromadb
from build_vector_store import LocalHuggingFaceEmbedding

def query_contract(question: str, k: int = 3):
    # 1. Connect to the existing Chroma store
    client = chromadb.PersistentClient(path="./chroma_store")

    # 2. Use the same embedding function as during indexing
    ef = LocalHuggingFaceEmbedding()
    collection = client.get_or_create_collection(
        name="loan_contract_chunks",
        embedding_function=ef
    )

    # 3. Query the vector store
    results = collection.query(
        query_texts=[question],
        n_results=k
    )

    # 4. Pretty-print the results
    docs = results["documents"][0]
    metas = results["metadatas"][0]
    dists = results["distances"][0]

    for i in range(len(docs)):
        print(f"\n--- RESULT {i+1} ---")
        print(f"Distance: {dists[i]:.4f}")
        print(f"Page: {metas[i].get('page_number')}")
        print(docs[i][:500], "...\n")  # first 500 chars of the chunk

if __name__ == "__main__":
    question = "What is the penalty for late payment?"
    query_contract(question, k=3)
