\# 🧠 Docu-Mind



\*\*AI-Powered Document Intelligence System using Retrieval-Augmented Generation (RAG)\*\*



\[!\[Python 3.11.9](https://img.shields.io/badge/Python-3.11.9-blue.svg)](https://www.python.org/downloads/)

\[!\[Streamlit](https://img.shields.io/badge/Streamlit-Latest-FF4B4B.svg)](https://streamlit.io/)

\[!\[LangChain](https://img.shields.io/badge/LangChain-0.1-green.svg)](https://www.langchain.com/)

\[!\[ChromaDB](https://img.shields.io/badge/ChromaDB-Vector\_Store-orange.svg)](https://www.trychroma.com/)

\[!\[License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)



> Transform complex documents into conversational insights. Upload PDFs, ask questions, get intelligent answers with precise citations.



\[Demo Video](#demo) • \[Features](#features) • \[Quick Start](#installation) • \[Tech Stack](#tech-stack)



---



\## 📋 Overview



Docu-Mind is a production-ready RAG (Retrieval-Augmented Generation) system that enables intelligent querying of PDF documents. Built for legal contracts, research papers, technical documentation, and business reports, it combines vector search with large language models to provide accurate, context-aware answers with source citations.



\*\*Key Capabilities:\*\*

\- 📄 Multi-document PDF processing with intelligent chunking

\- 🔍 Semantic search using vector embeddings

\- 💬 Natural language Q\&A with page-level citations

\- ⚡ Fast retrieval with ChromaDB vector store

\- 🎯 Context-aware responses using Groq's LLM API



---



\## ✨ Features



\### Core Functionality

\- \*\*Intelligent Document Parsing\*\*: Extracts and processes text from multi-page PDFs using `pdfplumber`

\- \*\*Smart Chunking\*\*: Splits documents into semantically meaningful chunks with overlap to preserve context

\- \*\*Vector Embeddings\*\*: Converts text chunks into high-dimensional vectors using Sentence Transformers

\- \*\*Semantic Search\*\*: Retrieves most relevant document sections using ChromaDB vector similarity

\- \*\*Context-Aware Answers\*\*: Generates precise responses using LangChain + Groq's Llama models

\- \*\*Source Attribution\*\*: Every answer includes page numbers and document references

\- \*\*Interactive UI\*\*: Clean Streamlit interface for document upload and real-time querying



\### Technical Highlights

\- \*\*Optimized Retrieval\*\*: Top-K similarity search with configurable relevance thresholds

\- \*\*Persistent Vector Store\*\*: ChromaDB local storage for fast repeated queries

\- \*\*Streaming Responses\*\*: Real-time answer generation with loading indicators

\- \*\*Error Handling\*\*: Graceful fallbacks for API failures and malformed documents



---



\## 🚀 Quick Start



\### Prerequisites

\- Python 3.11+

\- Groq API Key (\[Get one free](https://console.groq.com/))



\### Installation



```bash

\\# Clone the repository

git clone https://github.com/rishithreddy-16/docu-mind.git

cd docu-mind



\\# Create virtual environment

python -m venv venv

source venv/bin/activate  # On Windows: venv\\\\Scripts\\\\activate



\\# Install dependencies

pip install -r requirements.txt




