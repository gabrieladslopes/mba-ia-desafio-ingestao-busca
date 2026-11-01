import os
from dotenv import load_dotenv

from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from hashlib import sha256
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_postgres import PGVector
from time import sleep

PROJECT_ROOT = Path(__file__).parent.parent

load_dotenv(PROJECT_ROOT / ".env")

PDF_PATH = os.getenv("PDF_PATH")

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150
INGEST_BATCH_SIZE = 10
INGEST_BATCH_DELAY_SECONDS = 65

def ingest_pdf():
    required_variables = {
        "GOOGLE_API_KEY",
        "EMBEDDING_MODEL",
        "DATABASE_URL",
        "COLLECTION_NAME",
    }

    for variable in required_variables:
        value = os.getenv(variable)

        if not value or not value.strip():
            raise ValueError(f"Configure a variável {variable} no arquivo .env.")


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

    embeddings = GoogleGenerativeAIEmbeddings(
        model=os.environ["EMBEDDING_MODEL"],
    )

    print("Conectando ao armazenamento vetorial.")

    vector_store = PGVector(
        embeddings=embeddings,
        connection=os.environ["DATABASE_URL"],
        collection_name=os.environ["COLLECTION_NAME"],
        use_jsonb=True,
        create_extension=False
    )

    chunk_ids = []

    for index, chunk in enumerate(chunks):
        chunk.metadata["source"] = pdf_path.name
        chunk.metadata["chunk_index"] = index

        identity = (
            f"{os.environ['COLLECTION_NAME']}:"
            f"{pdf_path.name}:"
            f"{index}:"
            f"{chunk.page_content}"
        )

        chunk_id = sha256(identity.encode("utf-8")).hexdigest()
        chunk_ids.append(chunk_id)

    total_batches = (
        len(chunks) + INGEST_BATCH_SIZE - 1
    ) // INGEST_BATCH_SIZE

    processed_chunks = 0

    for batch_number, start in enumerate(
        range(0, len(chunks), INGEST_BATCH_SIZE),
        start=1,
    ):
        end = start + INGEST_BATCH_SIZE
        batch_chunks = chunks[start:end]
        batch_ids = chunk_ids[start:end]

        print(
            f"Processando lote {batch_number}/{total_batches}: {len(batch_chunks)} chunks.",
            flush=True
        )

        saved_ids = vector_store.add_documents(
            documents=batch_chunks,
            ids=batch_ids,
        )

        processed_chunks += len(saved_ids)

        print(
            f"Progresso: {processed_chunks}/{len(chunks)} chunks.",
            flush=True
        )

        if batch_number < total_batches:
            print(
                f"Aguardando {INGEST_BATCH_DELAY_SECONDS} segundos antes do próximo lote.",
                flush=True
            )

            sleep(INGEST_BATCH_DELAY_SECONDS)

    print(
        f"Ingestão concluída: {processed_chunks} chunks processados na coleção '{os.environ['COLLECTION_NAME']}'."
    )

if __name__ == "__main__":
    ingest_pdf()