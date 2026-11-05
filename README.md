# Ingestão e Busca Semântica com LangChain e Postgres

Projeto desenvolvido para o desafio técnico do MBA em Engenharia de Software com IA da FullCycle.

A aplicação lê um PDF, divide seu conteúdo em chunks, gera embeddings usando a Gemini API e armazena textos, vetores e metadados no PostgreSQL com pgVector.

O usuário pode fazer perguntas pelo terminal. A aplicação recupera os dez chunks mais próximos da pergunta e usa uma LLM para produzir uma resposta com base no contexto recuperado.

O prompt orienta o modelo a responder, quando não houver informação suficiente: "Não tenho informações necessárias para responder sua pergunta."

Repositório modelo: https://github.com/devfullcycle/mba-ia-desafio-ingestao-busca

## Tecnologias

- Python 3.12
- LangChain
- Google Gemini API
- PostgreSQL 17
- pgVector
- Docker e Docker Compose
- SQLAchemy e psycopg

As versões das dependências Python estão fixadas em 'requirements.txt'.

## Estrutura

- `docker-compose.yml`: banco PostgreSQL e habilitação da extensão vector
- `requirements.txt`: dependências Python
- `.env.example`: template de configuração, sem credenciais
- `src/ingest.py`: leitura, divisão e ingestão do PDF
- `src/search.py`: recuperação vetorial e construção da cadeia de resposta
- `src/chat.py`: interação pelo terminal
- `document.pdf`: documento utilizado na ingestão
- `README.md`: instruções de execução

## Funcionamento

### Ingestão

1. Carrega as páginas do PDF usando `PyPDFLoader`
2. Divide o texto usando `RecursiveCharacterTextSplitter`
3. Configura chunks de até 1000 caracteres e overlap de 150
4. Gera identificadores estáveis para os chunks
5. Consulta os identificadores já existentes na coleção
6. Gera embeddings somente dos chunks pendentes
7. Salva os chunks em lotes no PostgreSQL
8. Remove os registros desatualizados depois que todos os lotes terminam

O divisor procura respeitar separações naturais do texto. Por isso, os chunks podem ter menos de 1000 caracteres e a sobreposição efetiva pode variar. As páginas são divididas separadamente.

Para reduzir a ocorrência de limites da API, a ingestão utiliza lotes de 10 chunks e pausas de 65 segundos entre lotes.

### Perguntas

1. Recebe a pergunta pelo terminal
2. Gera seu embedding com o mesmo modelo usado na ingestão
3. Executa `similarity_search_with_score` com `k=10`
4. Concatena os textos recuperados
5. Preenche o prompt obrigatório da atividade
6. Envia o prompt à LLM
7. Exibe a resposta e aguarda a próxima pergunta

A cadeia é configurada uma vez na inicialização do chat e reutilizada nas perguntas seguintes.

## Pré-requisitos

- Python 3.12
- Git
- Docker
- Docker Compose
- Chave da Gemini API com acesso e cota para os modelos configurados

O ambiente foi validado em macOS com Python 3.12.2. Os comandos de ativação abaixo são para macOS e Linux.

## Preparação

### 1. Clone o repositório

```bash
git clone https://github.com/gabrieladslopes/mba-ia-desafio-ingestao-busca.git
cd mba-ia-desafio-ingestao-busca
```

### 2. Crie e ative o ambiente virtual

Crie o ambiente antes de instalar as dependências:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
```

Confira o interpretador:

```bash
python --version
command -v python
```

O caminho do Python deve apontar para `.venv/bin/python`.

Em um novo terminal, ative novamente o ambiente antes de executar os scripts:

```bash
source .venv/bin/activate
```

### 3. Instale as dependências

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip check
```

Resultado esperado da última verificação:

```text
No broken requirements found.
```

### 4. Configure as variáveis de ambiente

Copie o template:

```bash
cp .env.example .env
```

Edite `.env` e preencha a chave:

```dotenv
GOOGLE_API_KEY=sua_chave_aqui
EMBEDDING_MODEL=models/gemini-embedding-001
LLM_MODEL=gemini-3.5-flash-lite
PDF_PATH=document.pdf
DATABASE_URL=postgresql+psycopg://postgres:postgres@localhost:5432/rag
COLLECTION_NAME=mba_pdf
```

Crie a chave em: https://aistudio.google.com/api-keys

O arquivo `.env` contém a credencial local e deve permanecer ignorado pelo Git. O `.env.example` deve conter somente os nomes das variáveis e valores de exemplo, sem uma chave real.

Os nomes dos modelos, a disponibilidade e as cotas podem mudar. Consulte a documentação do provedor e valide o acesso antes da execução.

Caminhos relativos em `PDF_PATH` são interpretados a partir da raiz do projeto.

As credenciais do PostgreSQL acima correspondem ao ambiente local definido no Compose.

## Execução

Execute os comandos na raiz do projeto, com a `.venv` ativa.

### 1. Suba o banco

```bash
docker compose up -d
```

Confira os serviços:

```bash
docker compose ps -a
```

O serviço `postgres` deve ficar saudável (`healthy`). O serviço `bootstrap_vector_ext` executa a habilitação da extensão e deve terminar com código zero (`Exited (0)`).

Confirme a extensão no banco:

```bash
docker compose exec postgres psql -U postgres -d rag -c "SELECT extname, extversion FROM pg_extension WHERE extname = 'vector';"
```

A consulta deve retornar uma linha com `vector`. Se necessário, confira o bootstrap:

```bash
docker compose logs bootstrap_vector_ext
```

### 2. Execute a ingestão

```bash
python src/ingest.py
```

Com o PDF fornecido, foram observados:

- 34 páginas.
- 67 chunks.
- Maior chunk com 999 caracteres.
- Embeddings com 3.072 dimensões.

### 3. Inicie o chat

```bash
python src/chat.py
```

Exemplo de interação validada:

```text
Pergunta: Qual é o faturamento da Alfa Agronegócio Indústria?
Resposta: R$ 85.675.568,77

Pergunta: Em que ano foi fundada a Alfa Agronegócio Indústria?
Resposta: A Alfa Agronegócio Indústria foi fundada em 1931.

Pergunta: Qual é a capital da França?
Resposta: Não tenho informações necessárias para responder sua pergunta.
```

Digite `sair` ou pressione Control + C para encerrar. Perguntas vazias são rejeitadas antes da chamada à cadeia.

## Problemas comuns

### Erro 429 da Gemini API

Confira a mensagem para identificar a cota atingida. Os limites podem envolver tokens por minuto, requisições por minuto ou requisições por dia.

A ingestão já usa pausas entre lotes. Ainda assim, outras chamadas no mesmo projeto podem consumir a cota compartilhada.

Se a falha for temporária, aguarde a liberação da cota e execute novamente. Os chunks já gravados serão reaproveitados.

### Falha de conexão com PostgreSQL

Confira:

```bash
docker compose ps -a
```

Verifique também se `DATABASE_URL` corresponde ao usuário, senha, banco e porta definidos no Compose.

### Porta 5432 ocupada

Outro serviço pode estar usando a porta. Se alterar a porta publicada no Compose, ajuste também a porta de `DATABASE_URL`.

### Extensão vector ausente

Confira os logs do serviço de inicialização:

```bash
docker compose logs bootstrap_vector_ext
```

Depois confirme a extensão pela consulta apresentada nas instruções de execução.

### Logs de ALTS do gRPC

Na combinação de dependências utilizada, podem aparecer mensagens como:

```text
ALTS creds ignored. Not running on GCP...
```

Nos testes realizados, essas mensagens não impediram a ingestão, as respostas ou o encerramento. Falhas acompanhadas de traceback ou erro de API devem ser investigadas separadamente.

## Encerramento do ambiente

Para parar os serviços:

```bash
docker compose stop
```

Para remover os contêineres e a rede do Compose, preservando o volume de dados:

```bash
docker compose down
```

Não use `docker compose down -v` se quiser manter os dados: a opção `-v` remove o volume do banco deste projeto.

Para sair do ambiente virtual:

```bash
deactivate
```