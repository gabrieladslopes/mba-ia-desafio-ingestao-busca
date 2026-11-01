import os
from dotenv import load_dotenv

from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

PROJECT_ROOT = Path(__file__).parent.parent

load_dotenv(PROJECT_ROOT / ".env")

PDF_PATH = os.getenv("PDF_PATH")

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150

def ingest_pdf():
    if not PDF_PATH or not PDF_PATH.strip():
        raise ValueError("Configure a variável PDF_PATH no arquivo .env.");

    pdf_path = Path(PDF_PATH);

    if not pdf_path.is_absolute():
        pdf_path = PROJECT_ROOT / pdf_path

    if not pdf_path.is_file():
        raise FileNotFoundError(f"Arquivo PDF não encontrado: {pdf_path}")

    print(f"Lendo PDF: {pdf_path.name}")

    loader = PyPDFLoader(str(pdf_path))
    documents = loader.load()

    if not any(document.page_content.strip() for document in documents):
        raise ValueError("O PDF está vazio ou não contém texto legível.")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len
    )

    chunks = splitter.split_documents(documents)

    if not chunks:
        raise ValueError("Nenhum chunk foi gerado a partir do PDF.")

    largest_chunk = max(len(chunk.page_content) for chunk in chunks)

    print(f"Páginas carregadas: {len(documents)}")
    print(f"Chunks gerados: {len(chunks)}")
    print(f"Tamanho do maior chunk: {largest_chunk} caracteres")

    first_chunk = chunks[0]

    print("Primeiro chunk:")
    print(first_chunk.page_content)

    print("Metadados do primeiro chunk:")
    print(first_chunk.metadata)

if __name__ == "__main__":
    ingest_pdf()