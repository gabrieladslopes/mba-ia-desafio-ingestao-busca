import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_google_genai import ( ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings )
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
    llm_model = os.getenv("LLM_MODEL")

    if not llm_model or not llm_model.strip():
        raise ValueError("Configure a variável LLM_MODEL no arquivo .env.")

    vector_store = create_vector_store()

    llm = ChatGoogleGenerativeAI(
        model=llm_model,
        max_tokens=1024,
        timeout=60,
        max_retries=0,
    )

    prompt = PromptTemplate.from_template(PROMPT_TEMPLATE)

    def prepare_context(question):
        if not isinstance(question, str) or not question.strip():
            raise ValueError("Informe uma pergunta não vazia.")

        question = question.strip()

        results = search_documents(
            question=question,
            vector_store=vector_store
        )

        context = '\n\n'.join(
            document.page_content
            for document, distance in results
        )

        return {
            "contexto": context,
            "pergunta": question
        }

    chain = (
        RunnableLambda(prepare_context)
        | prompt
        | llm
        | StrOutputParser()
    )

    return chain