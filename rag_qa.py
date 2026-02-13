# rag_qa.py - PRODUCTION-READY VERSION WITH ENHANCEMENTS

import os
import numpy as np
import streamlit as st
from typing import List, Dict

from langchain_groq import ChatGroq
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.embeddings import Embeddings
from langchain_core.documents import Document
from sentence_transformers import SentenceTransformer

# ============ CONFIG ============
os.environ["GROQ_API_KEY"] = st.secrets.get("GROQ_API_KEY", "")

CHROMA_DIR = "./chroma_store"
COLLECTION_NAME = "loan_contract_chunks"
# ================================


class LocalHuggingFaceEmbedding(Embeddings):
    """LangChain-compatible embedding class using SentenceTransformer"""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed multiple documents (returns 2D list)"""
        embeddings = self.model.encode(texts)
        return embeddings.tolist()

    def embed_query(self, text: str) -> List[float]:
        """Embed single query (returns 1D list)"""
        embedding = self.model.encode(text)
        return embedding.tolist()


def smart_retrieval(vectorstore, question: str, k: int = 10) -> List[Document]:
    """
    Smart retrieval with question-type awareness.

    Strategy:
    - Party questions → prioritize early pages (1–10)
    - Definition-style questions → broader search (more chunks)
    - Otherwise → standard semantic search
    """
    question_lower = question.lower()

    # Category 1: Party identification (usually in first 10 pages)
    party_keywords = ["who is", "lender", "borrower", "parties", "party",
                  "administrative agent", "collateral agent", "guarantor"]
    
    if any(keyword in question_lower for keyword in party_keywords):
        print(" Party question detected - prioritizing early pages")

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

    # Category 2: Definition / "what is" questions
    definition_keywords = ["definition", "define", "what does", "meaning of", "what is"]
    if any(keyword in question_lower for keyword in definition_keywords):
        print("   📖 Definition question detected - searching broader context")
        return vectorstore.similarity_search(question, k=k + 5)

    # Category 3: Default behavior
    return vectorstore.similarity_search(question, k=k)


def load_vectorstore():
    """Load vectorstore (Chroma) with local embeddings"""
    print("🔧 Loading retriever...")
    embedding_function = LocalHuggingFaceEmbedding()

    vectorstore = Chroma(
        persist_directory=CHROMA_DIR,
        embedding_function=embedding_function,
        collection_name=COLLECTION_NAME
    )

    print("✅ Retriever loaded successfully")
    return vectorstore


def create_rag_chain():
    """Build enhanced RAG pipeline with smart retrieval + strict prompt"""
    vectorstore = load_vectorstore()

    print("🔧 Initializing Groq LLM...")
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
        """Format docs with smart retrieval and deduplication."""
        question = input_dict["question"]

        docs = smart_retrieval(vectorstore, question, k=12)

        print(f"\n📄 Retrieved {len(docs)} chunks:")
        for i, doc in enumerate(docs[:5], 1):
            page = doc.metadata.get("page_number", "Unknown")
            preview = doc.page_content[:80].replace("\n", " ")
            print(f"  {i}. Page {page}: {preview}...")
        if len(docs) > 5:
            print(f"  ... and {len(docs) - 5} more chunks")

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


def interactive_mode():
    """Run interactive Q&A session with light conversation history."""
    print("=" * 70)
    print("📄 Docu-Mind - Interactive Contract Q&A")
    print("=" * 70)

    # Diagnostics
    print("\n🔍 Running diagnostics...")
    try:
        ef = LocalHuggingFaceEmbedding()
        q_emb = ef.embed_query("Test query")
        d_emb = ef.embed_documents(["Test doc 1", "Test doc 2"])
        q_shape = np.array(q_emb).shape
        d_shape = np.array(d_emb).shape
        assert len(q_shape) == 1
        assert len(d_shape) == 2
        print("✅ Embeddings working correctly")
    except Exception as e:
        print(f"❌ Diagnostic failed: {e}")
        return

    # Load RAG system
    print("\n🚀 Initializing RAG system...")
    try:
        qa_chain = create_rag_chain()
        print("✅ System ready!\n")
    except Exception as e:
        print(f"❌ Initialization failed: {e}")
        return

    # Conversation history: last few Q&A turns
    conversation_history: List[Dict[str, str]] = []

    print("💡 Sample Questions:")
    sample_qs = [
        "Who is the lender?",
        "What is the interest rate?",
        "What is the penalty for late payment?",
        "When does the loan mature?",
        "What are the events of default?"
    ]
    for q in sample_qs:
        print(f"   • {q}")

    print("\nType your question (or 'exit' to quit)\n")
    print("=" * 70 + "\n")

    while True:
        try:
            user_q = input("💬 Your Question: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n\n👋 Thank you for using Docu-Mind!")
            break

        if user_q.lower() in ["exit", "quit", "q", "bye", "stop"]:
            print("\n👋 Thank you for using Docu-Mind!")
            break

        if not user_q:
            print("⚠️  Please enter a question.\n")
            continue

        # Build lightweight conversation-aware question
        if conversation_history:
            recent = conversation_history[-2:]
            prev_blocks = []
            for h in recent:
                prev_blocks.append(
                    f"Previous question: {h['question']}\n"
                    f"Previous answer (truncated): {h['answer'][:180]}..."
                )
            history_text = "\n\n".join(prev_blocks)
            question_for_model = f"{history_text}\n\nCurrent question: {user_q}"
        else:
            question_for_model = user_q

        print("\n🔍 Searching contract...")
        try:
            answer = qa_chain.invoke({"question": question_for_model})

            print("\n✅ Answer:")
            print("-" * 70)
            print(answer)
            print("-" * 70 + "\n")

            conversation_history.append({
                "question": user_q,
                "answer": answer
            })

        except Exception as e:
            print(f"\n❌ Error: {e}\n")


def batch_mode():
    """Run predefined batch of questions"""
    print("=" * 70)
    print("📄 Docu-Mind - Batch Question Answering")
    print("=" * 70 + "\n")

    try:
        qa_chain = create_rag_chain()

        questions = [
            "Who is the lender?",
            "What is the interest rate?",
            "What is the penalty for late payment?",
            "When does the loan mature?",
            "What are the prepayment terms?"
        ]

        for i, q in enumerate(questions, 1):
            print(f"\n{'=' * 70}")
            print(f"Question {i}/{len(questions)}: {q}")
            print("=" * 70)

            answer = qa_chain.invoke({"question": q})

            print("\n✅ Answer:")
            print("-" * 70)
            print(answer)
            print("-" * 70)

    except Exception as e:
        print(f"❌ Error: {e}")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "batch":
        batch_mode()
    else:
        interactive_mode()

