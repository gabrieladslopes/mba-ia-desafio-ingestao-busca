import os
from dotenv import load_dotenv

from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from hashlib import sha256
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_postgres import PGVector
from time import sleep
from sqlalchemy import create_engine, text

PROJECT_ROOT = Path(__file__).resolve().parent.parent

load_dotenv(PROJECT_ROOT / ".env")

PDF_PATH = os.getenv("PDF_PATH")

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150
INGEST_BATCH_SIZE = 10
INGEST_BATCH_DELAY_SECONDS = 65

def get_existing_chunk_ids(database_url, collection_name):
    engine = create_engine(database_url)

    query = text("""
        SELECT e.id
        FROM langchain_pg_embedding AS e
        JOIN langchain_pg_collection AS c
            ON c.uuid = e.collection_id
        WHERE c.name = :collection_name
    """)

    try:
        with engine.connect() as connection:
            result = connection.execute(
                query,
                {"collection_name": collection_name},
            )
            return set(result.scalars().all())
    finally:
        engine.dispose()

def ingest_pdf():
    required_variables = (
        "GOOGLE_API_KEY",
        "EMBEDDING_MODEL",
        "DATABASE_URL",
        "COLLECTION_NAME",
        "PDF_PATH"
    )

    for variable in required_variables:
        value = os.getenv(variable)

        if not value or not value.strip():
            raise ValueError(f"Configure a variável {variable} no arquivo .env.")

    pdf_path = Path(os.environ["PDF_PATH"]);

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

    existing_ids = get_existing_chunk_ids(
        database_url=os.environ["DATABASE_URL"],
        collection_name=os.environ["COLLECTION_NAME"]
    )

    current_ids = set(chunk_ids)

    pending_chunks = []
    pending_ids = []

    for chunk, chunk_id in zip(chunks, chunk_ids):
        if chunk_id not in existing_ids:
            pending_chunks.append(chunk)
            pending_ids.append(chunk_id)

    reused_chunks = len(chunks) - len(pending_chunks)

    print(f"Chunks reaproveitados: {reused_chunks}")
    print(f"Chunks pendentes: {len(pending_chunks)}")

    total_batches = (
        len(pending_chunks) + INGEST_BATCH_SIZE - 1
    ) // INGEST_BATCH_SIZE

    processed_chunks = 0

    for batch_number, start in enumerate(
        range(0, len(pending_chunks), INGEST_BATCH_SIZE),
        start=1,
    ):
        end = start + INGEST_BATCH_SIZE

        batch_chunks = pending_chunks[start:end]
        batch_ids = pending_ids[start:end]

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
            f"Progresso: {processed_chunks}/{len(pending_chunks)} chunks pendentes.",
            flush=True
        )

        if batch_number < total_batches:
            print(
                f"Aguardando {INGEST_BATCH_DELAY_SECONDS} segundos antes do próximo lote.",
                flush=True
            )

            sleep(INGEST_BATCH_DELAY_SECONDS)

    obsolete_ids = existing_ids - current_ids

    if obsolete_ids:
        print(f"Removendo {len(obsolete_ids)} chunks desatualizados...")

        vector_store.delete(
            ids=list(obsolete_ids),
            collection_only=True
        )

    print(
        f"Ingestão concluída: {len(chunks)} chunks na versão atual; "
        f"{reused_chunks} chunks reaproveitados; " 
        f"{processed_chunks} chunks adicionados; "
        f"{len(obsolete_ids)} chunks removidos."
    )

if __name__ == "__main__":
    ingest_pdf()