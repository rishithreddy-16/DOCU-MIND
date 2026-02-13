import pdfplumber
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


# 1. THE EXTRACTION FUNCTION
def extract_text_with_metadata(pdf_path):
    print(f"🔍 Extracting text from {pdf_path}...")
    documents = []
    skipped_pages = []
    
    with pdfplumber.open(pdf_path) as pdf:
        total_pages = len(pdf.pages)
        print(f"📄 Total Pages Found in PDF: {total_pages}")
        
        for i, page in enumerate(pdf.pages):
            # Try 1: Standard extraction
            text = page.extract_text()

            if text and text.strip():  # Check if text is not just empty space
                doc = Document(
                    page_content=text,
                    metadata={"page_number": i + 1, "source": pdf_path}
                )
                documents.append(doc)
            else:
                skipped_pages.append(i + 1)
                
    print(f"✅ Extracted {len(documents)} pages.")
    if skipped_pages:
        print(f"⚠️ Skipped {len(skipped_pages)} pages (Blank or Images): {skipped_pages}")
        
    return documents


# 2. THE CHUNKING FUNCTION
def chunk_documents(documents):
    print("✂️ Chunking documents...")
    
    # Why 1000? It's roughly 2-3 paragraphs. Good for retrieving specific clauses.
    # Why overlap 200? If a sentence gets cut in half, the next chunk picks it up.
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000,
        chunk_overlap=300,
        separators=["\n\n", "\n", " ", ""] # Try to split by paragraphs first
    )
    
    chunks = text_splitter.split_documents(documents)
    print(f"✅ Split {len(documents)} pages into {len(chunks)} chunks.")
    return chunks

# 3. MAIN EXECUTION
if __name__ == "__main__":
    # Load the PDF
    raw_docs = extract_text_with_metadata("C:\\Users\\rishi\\Downloads\\projects\\Docu-Mind\\contract.pdf")
    
    # Split into chunks
    final_chunks = chunk_documents(raw_docs)
    
    # Inspect the first chunk to see if it worked
    print("\n--- PREVIEW OF CHUNK 1 ---")
    print(final_chunks[0].page_content)
    print("\n--- METADATA ---")
    print(final_chunks[0].metadata)

# ingest.py

def load_and_chunk(pdf_path: str):
    raw_docs = extract_text_with_metadata(pdf_path)
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    chunks = splitter.split_documents(raw_docs)
    return chunks
