import os
import shutil
from typing import List, Dict

import streamlit as st
from langchain_groq import ChatGroq
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.embeddings import Embeddings  # LangChain interface
from langchain_core.documents import Document
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.api.types import EmbeddingFunction, Documents, Embeddings as ChromaEmbeddings

from ingest import load_and_chunk

# ============ CONFIG ============
os.environ["GROQ_API_KEY"] = st.secrets.get("GROQ_API_KEY", "")

CHROMA_DIR = "./chroma_store"
COLLECTION_NAME = "loan_contract_chunks"
# ================================

# ============ UNIFIED EMBEDDING CLASS ============
class LocalHuggingFaceEmbedding(Embeddings):
    """
    Unified embedding class that implements BOTH:
    - LangChain's Embeddings interface (embed_documents, embed_query)
    - ChromaDB's EmbeddingFunction interface (__call__)
    """
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

    # LangChain interface
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed multiple documents (returns 2D list)"""
        embeddings = self.model.encode(texts)
        return embeddings.tolist()

    def embed_query(self, text: str) -> List[float]:
        """Embed single query (returns 1D list)"""
        embedding = self.model.encode(text)
        return embedding.tolist()

    # ChromaDB interface (for raw chromadb usage)
    def __call__(self, input: Documents) -> ChromaEmbeddings:
        """ChromaDB compatibility"""
        embeddings = self.model.encode(input)
        return embeddings.tolist()
# ================================================

def smart_retrieval(vectorstore, question: str, k: int = 10) -> List[Document]:
    """Smart retrieval with question-type awareness."""
    question_lower = question.lower().strip()

    # 1) Definition / "what is" style questions
    definition_keywords = ["definition", "define", "what does", "meaning of", "what is"]
    if any(keyword in question_lower for keyword in definition_keywords):
        return vectorstore.similarity_search(question, k=k + 5)

    # 2) Party identification questions
    if question_lower.startswith("who is") or question_lower.startswith("who are"):
        early_docs = vectorstore.similarity_search(
            question,
            k=k // 2,
            filter={"page_number": {"$lte": 10}}
        )
        other_docs = vectorstore.similarity_search(question, k=k // 2)

        seen_content = set()
        combined_docs: List[Document] = []

        for doc in early_docs + other_docs:
            content_hash = doc.page_content[:100]
            if content_hash not in seen_content:
                seen_content.add(content_hash)
                combined_docs.append(doc)

        return combined_docs[:k]

    # 3) Default behavior
    return vectorstore.similarity_search(question, k=k)


@st.cache_resource(show_spinner=False)
def load_vectorstore() -> Chroma:
    """Load existing Chroma store with embeddings (cached)."""
    ef = LocalHuggingFaceEmbedding()
    vs = Chroma(
        persist_directory=CHROMA_DIR,
        embedding_function=ef,
        collection_name=COLLECTION_NAME
    )
    return vs


@st.cache_resource(show_spinner=False)
def load_llm_chain():
    """Create and cache the RAG chain."""
    vectorstore = load_vectorstore()

    llm = ChatGroq(
        model="llama-3.1-8b-instant",
        temperature=0.0,
        timeout=30.0,
        max_retries=2
    )

    prompt = ChatPromptTemplate.from_template("""
You are a legal document assistant analyzing a loan contract.

STRICT RULES:
1. Answer ONLY using explicit information from the Context below.
2. If the answer is not clearly stated, respond: "I cannot find that information in the provided contract sections".
3. Always cite page numbers in format: (Page X).
4. For party identification questions:
   - Provide the full legal entity name.
   - Include any additional identifiers (address, registration number if stated).
   - Cite the exact section (e.g., "Preamble, Page 1" or "Section X.XX").
5. Be concise - 2–3 sentences maximum unless more detail is explicitly requested.

Context:
{context}

Question: {question}

Answer:""")

    def smart_format_docs(input_dict: Dict[str, str]) -> Dict[str, str]:
        question = input_dict["question"]
        docs = smart_retrieval(vectorstore, question, k=12)

        context_parts = []
        for doc in docs:
            page = doc.metadata.get("page_number", "Unknown")
            content = doc.page_content.strip()
            context_parts.append(f"[Page {page}]\n{content}")

        full_context = "\n\n".join(context_parts)

        return {
            "context": full_context,
            "question": question
        }

    chain = (
        RunnablePassthrough()
        | smart_format_docs
        | prompt
        | llm
        | StrOutputParser()
    )

    return chain


def rebuild_index_from_pdf(uploaded_file) -> int:
    """Save uploaded PDF and rebuild Chroma index. Returns chunk count."""
    # Clean old store
    if os.path.exists(CHROMA_DIR):
        shutil.rmtree(CHROMA_DIR)

    os.makedirs(CHROMA_DIR, exist_ok=True)

    # Save uploaded file
    pdf_path = os.path.join(CHROMA_DIR, "uploaded_contract.pdf")
    with open(pdf_path, "wb") as f:
        f.write(uploaded_file.getvalue())

    # Chunk PDF
    chunks = load_and_chunk(pdf_path)

    # Build collection using raw ChromaDB
    chroma_client = chromadb.PersistentClient(path=CHROMA_DIR)
    
    try:
        chroma_client.delete_collection(name=COLLECTION_NAME)
    except:
        pass

    ef = LocalHuggingFaceEmbedding()
    collection = chroma_client.create_collection(
        name=COLLECTION_NAME,
        embedding_function=ef
    )

    documents = [c.page_content for c in chunks]
    metadatas = [c.metadata for c in chunks]
    ids = [f"chunk-{i}" for i in range(len(chunks))]

    collection.add(ids=ids, documents=documents, metadatas=metadatas)

    # Clear Streamlit caches
    load_vectorstore.clear()
    load_llm_chain.clear()

    return len(chunks)


# =============== STREAMLIT UI ===============

st.set_page_config(
    page_title="Docu-Mind",
    page_icon="📄",
    layout="wide"
)

st.title("📄 Docu-Mind")
st.subheader("Ask precise questions about your loan contract and get answers with page citations.")

with st.sidebar:
    st.header("📁 Contract Upload")

    uploaded_file = st.file_uploader("Upload a loan contract PDF", type=["pdf"])
    if uploaded_file is not None:
        if st.button("Process PDF"):
            with st.spinner("Processing and indexing PDF..."):
                try:
                    chunk_count = rebuild_index_from_pdf(uploaded_file)
                    st.success(f"✅ Indexed {chunk_count} chunks from the uploaded contract.")
                    st.session_state["contract_ready"] = True
                except Exception as e:
                    st.error(f"❌ Failed to process PDF: {e}")
                    import traceback
                    st.code(traceback.format_exc())
    else:
        st.info("Upload a PDF and click 'Process PDF' to start.")

    st.markdown("---")
    st.markdown("**Status:**")
    if st.session_state.get("contract_ready", False):
        st.success("Contract loaded ✅")
    else:
        st.warning("No contract loaded ⚠️")

    st.markdown("---")
    st.markdown("**💡 Sample Questions:**")
    st.caption("• Who is the lender?")
    st.caption("• What is the interest rate?")
    st.caption("• What is the penalty for late payment?")
    st.caption("• When does the loan mature?")

    st.markdown("---")
    st.caption("Powered by Groq + SentenceTransformers + Chroma")

# Chat state
if "messages" not in st.session_state:
    st.session_state["messages"] = []

st.markdown("### 💬 Chat with your contract")

# Display chat history
for msg in st.session_state["messages"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Chat input
user_input = st.chat_input("Type your question about the contract...")

if user_input:
    if not st.session_state.get("contract_ready", False):
        st.warning("⚠️ Please upload and process a contract PDF first.")
    else:
        # Add user message
        st.session_state["messages"].append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        # Build conversation-aware question
        history_text = ""
        history_slice = [m for m in st.session_state["messages"] if m["role"] == "user"][:-1]
        if history_slice:
            recent = history_slice[-2:]
            blocks = [f"Previous question: {m['content']}" for m in recent]
            history_text = "\n\n".join(blocks)

        if history_text:
            question_for_model = f"{history_text}\n\nCurrent question: {user_input}"
        else:
            question_for_model = user_input

        # Get answer
        with st.chat_message("assistant"):
            with st.spinner("🔍 Searching contract..."):
                try:
                    qa_chain = load_llm_chain()
                    answer = qa_chain.invoke({"question": question_for_model})
                    st.markdown(answer)
                    st.session_state["messages"].append(
                        {"role": "assistant", "content": answer}
                    )
                except Exception as e:
                    st.error(f"❌ Error: {e}")
                    import traceback
                    st.code(traceback.format_exc())

