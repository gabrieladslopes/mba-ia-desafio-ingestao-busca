import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_postgres import PGVector

PROJECT_ROOT = Path(__file__).resolve().parent.parent

load_dotenv(PROJECT_ROOT / ".env")

SEARCH_RESULTS = 10

PROMPT_TEMPLATE = """
CONTEXTO:
{contexto}

REGRAS:
- Responda somente com base no CONTEXTO.
- Se a informação não estiver explicitamente no CONTEXTO, responda:
  "Não tenho informações necessárias para responder sua pergunta."
- Nunca invente ou use conhecimento externo.
- Nunca produza opiniões ou interpretações além do que está escrito.

EXEMPLOS DE PERGUNTAS FORA DO CONTEXTO:
Pergunta: "Qual é a capital da França?"
Resposta: "Não tenho informações necessárias para responder sua pergunta."

Pergunta: "Quantos clientes temos em 2024?"
Resposta: "Não tenho informações necessárias para responder sua pergunta."

Pergunta: "Você acha isso bom ou ruim?"
Resposta: "Não tenho informações necessárias para responder sua pergunta."

PERGUNTA DO USUÁRIO:
{pergunta}

RESPONDA A "PERGUNTA DO USUÁRIO"
"""

def create_vector_store():
    required_variables = (
        "GOOGLE_API_KEY",
        "EMBEDDING_MODEL",
        "DATABASE_URL",
        "COLLECTION_NAME",
    )

    for variable in required_variables:
        value = os.getenv(variable)

        if not value or not value.strip():
            raise ValueError(f"Configure a variável {variable} no arquivo .env.")

    embeddings = GoogleGenerativeAIEmbeddings(
        model=os.environ["EMBEDDING_MODEL"],
    )

    return PGVector(
        embeddings=embeddings,
        connection=os.environ["DATABASE_URL"],
        collection_name=os.environ["COLLECTION_NAME"],
        use_jsonb=True,
        create_extension=False,
    )

def search_documents(question, vector_store):
    if not isinstance(question, str) or not question.strip():
        raise ValueError("Informe uma pergunta não vazia.")

    return vector_store.similarity_search_with_score(
        query=question.strip(),
        k=SEARCH_RESULTS,
    )

def search_prompt(question=None, vector_store=None):
    if not isinstance(question, str) or not question.strip():
        raise ValueError("Informe uma pergunta não vazia.")

    if vector_store is None:
        vector_store = create_vector_store()
    
    results = search_documents(question, vector_store)

    context = "\n".join(document.page_content for document, distance in results)

    return PROMPT_TEMPLATE.format(contexto=context, pergunta=question.strip())

def main():
    question = input("Digite uma pergunta para testar a busca: ").strip()

    if not question:
        print("Nenhuma pergunta informada.")
        return

    vector_store = create_vector_store()

    print("Buscando os trechos mais próximos.")
    results = search_documents(question, vector_store)

    print(f"Resultados recuperados: {len(results)}")
    
    for position, (document, distance) in enumerate(results, start=1):
        page = document.metadata.get("page")
        page_display = page + 1 if isinstance(page, int) else "não informada"

        print(f"\n--- Resultado {position} ---")
        print(f"Distância: {distance:.4f}")
        print(f"Página: {page_display}")
        print(document.page_content)

if __name__ == "__main__":
    main()