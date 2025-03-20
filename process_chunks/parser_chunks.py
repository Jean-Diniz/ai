"""
Exemplo de pipeline para:
  - Conexão com SQLite para controle de arquivos processados;
  - Acesso ao Google Drive para listagem e download de arquivos Markdown e PDF;
  - Extração de metadados e keywords utilizando LLMs via LangChain;
  - Processamento de documentos e inserção em um índice no Pinecone.
  
Certifique-se de configurar as variáveis de ambiente (via .env) e instalar as dependências para as chaves de API e credenciais.
"""

from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
import os
import json
import io
import sqlite3
from pinecone import Pinecone
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor, as_completed
from uuid import uuid4

# Carrega variáveis de ambiente (ex: GEMINI_API_KEY, GOOGLE_APPLICATION_CREDENTIALS, DRIVE_FOLDER_ID)
load_dotenv(override=True)

# Acesso ao Google Drive
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload

# Integração com LLMs (exemplo usando ChatGoogleGenerativeAI do LangChain)
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_pinecone import PineconeVectorStore
from langchain.docstore.document import Document
from langchain_core.output_parsers import PydanticOutputParser

# Validação dos dados extraídos
from pydantic import BaseModel, Field, model_validator
from typing import List, Optional, Union


# Modelo para extração de keywords
class KeywordsOutput(BaseModel):
    palavras_chave: List[str] = Field(..., description="Lista de palavras-chave extraídas")
    
    @model_validator(mode='before')
    def ajustar_chave(cls, values):
        if "palavras-chave" in values:
            values["palavras_chave"] = values.pop("palavras-chave")
        return values


# Modelo para metadados gerais dos documentos
class GeneralMetadata(BaseModel):
    titulo_obra: str = Field(..., description="Título da obra")
    autor: Optional[str] = Field(..., description="Nome do(s) autor(es)")
    bioma: Optional[Union[str, List[str]]] = Field(
        None,
        description=("Identifique o bioma (Amazônia, Cerrado, Caatinga, Mata Atlântica, Pampa ou Pantanal). "
                     "Caso não aplicável, deixe em branco.")
    )


def log_retry(retry_state):
    print(f"Última exceção: {retry_state.outcome.exception()}")
    print(f"Tentativa {retry_state.attempt_number} falhou. Tentando em {retry_state.idle_for} segundos.")

embeddings = GoogleGenerativeAIEmbeddings(
    api_key=os.getenv("GEMINI_API_KEY"),
    model="models/text-embedding-004"
)

# -----------------------------------------------------------
# 1. Conexão com banco de dados (SQLite)
# -----------------------------------------------------------
def get_db_connection(db_path='files_metadata.db'):
    """Retorna conexão com o banco SQLite."""
    conn = sqlite3.connect(db_path)
    return conn


def create_table_if_not_exists(conn):
    """Cria a tabela de controle se ainda não existir."""
    create_table_sql = """
    CREATE TABLE IF NOT EXISTS files_upload_control (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        file_id TEXT UNIQUE,
        file_name TEXT
    );
    """
    with conn:
        conn.execute(create_table_sql)


def get_processed_file_ids(conn):
    """Retorna um conjunto de IDs dos arquivos já processados."""
    query = "SELECT file_id FROM files_upload_control;"
    rows = conn.execute(query).fetchall()
    return set(row[0] for row in rows)


def mark_file_as_processed(conn, file_id, file_name):
    """Marca o arquivo como processado inserindo seu registro no banco."""
    insert_sql = """
    INSERT OR IGNORE INTO files_upload_control (file_id, file_name)
    VALUES (?, ?);
    """
    with conn:
        conn.execute(insert_sql, (file_id, file_name))


# -----------------------------------------------------------
# 2. Conexão com Google Drive (listagem e download)
# -----------------------------------------------------------
def get_drive_service():
    """
    Retorna o serviço do Google Drive utilizando Service Account.
    Certifique-se de que o arquivo de credenciais está configurado corretamente.
    """
    SERVICE_ACCOUNT_FILE = os.getenv("GOOGLE_APPLICATION_CREDENTIALS", "service_account.json")
    SCOPES = ["https://www.googleapis.com/auth/drive"]
    
    credentials = service_account.Credentials.from_service_account_file(
        SERVICE_ACCOUNT_FILE, scopes=SCOPES
    )
    service = build("drive", "v3", credentials=credentials)
    return service


def list_markdown_files(drive_service, folder_id):
    """
    Lista arquivos Markdown (.md) em uma pasta do Google Drive.
    
    Retorna:
        Lista de dicionários com 'id', 'name' e 'mimeType'.
    """
    query = (
        f"'{folder_id}' in parents "
        "and mimeType = 'text/markdown' "
        "and trashed = false"
    )
    results = drive_service.files().list(
        pageSize=1000,
        q=query,
        fields="files(id, name, mimeType)"
    ).execute()
    return results.get('files', [])


def download_file_as_text(drive_service, file_id):
    """
    Faz download do arquivo e retorna o conteúdo em texto.
    """
    request = drive_service.files().get_media(fileId=file_id)
    fh = io.BytesIO()
    downloader = MediaIoBaseDownload(fh, request)
    
    done = False
    while not done:
        status, done = downloader.next_chunk()
    
    fh.seek(0)
    return fh.read().decode('utf-8', errors='ignore')


def find_pdf_with_same_name(drive_service, folder_id, base_name):
    """
    Busca um PDF na mesma pasta com nome semelhante ao do arquivo Markdown.
    
    Retorna:
        Primeiro PDF encontrado ou None.
    """
    pdf_name = base_name + ".pdf"
    query = (
        f"'{folder_id}' in parents "
        f"and name = '{pdf_name}' "
        "and mimeType = 'application/pdf' "
        "and trashed = false"
    )
    results = drive_service.files().list(
        q=query,
        fields="files(id, name, mimeType)"
    ).execute()
    files = results.get('files', [])
    return files[0] if files else None


# -----------------------------------------------------------
# 3. Processamento de texto e chamadas à LLM (LangChain)
# -----------------------------------------------------------
def extract_text_from_file(content_str):
    """
    Processa o texto bruto removendo trechos indesejados.
    """
    return content_str.replace("<!-- image -->\n\n", "").replace("<!-- image -->", "")


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=60),
    retry=retry_if_exception_type(Exception),
    after=log_retry
)
def call_llm_extract_general_metadata(text):
    """
    Chama a LLM para extrair metadados gerais (título, autor, bioma).
    
    Retorna:
        Objeto GeneralMetadata ou dicionário vazio em caso de erro.
    """
    llm = ChatGoogleGenerativeAI(
        api_key=os.getenv("GEMINI_API_KEY"),
        model="models/gemini-1.5-flash",
        temperature=0.7
    )
    system_prompt = (
        "Você é um algoritmo especialista em extração.\n"
        "Extraia apenas as informações relevantes do texto.\n"
        "Se não souber o valor de um atributo solicitado, pode omitir seu valor."
    )
    prompt = f"""{system_prompt}
    
Se o texto contiver múltiplos artigos, extraia os metadados da editora e retorne apenas 1 conjunto de metadados.

Texto:
{text}

Atributos para extrair (em JSON):
- titulo_obra: (string)
- autor: (string)
- bioma: (string) [Amazônia, Cerrado, Caatinga, Mata Atlântica, Pampa ou Pantanal]

Retorne somente o JSON no formato:
{{
  "titulo_obra": "...",
  "autor": "...",
  "bioma": "..."
}}
"""
    response = llm.invoke(prompt)
    parser = PydanticOutputParser(pydantic_object=GeneralMetadata)
    try:
        return parser.parse(response.text())
    except Exception as e:
        print("Erro no parse:", e)
        return {}


def split_text_into_chunks(text, chunk_size=512, chunk_overlap=100):
    """
    Divide o texto em chunks utilizando RecursiveCharacterTextSplitter, evitando fragmentar tabelas em Markdown.
    
    Retorna:
        Lista de objetos Document.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", "."]
    )
    return splitter.create_documents([text])


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=60),
    retry=retry_if_exception_type(Exception),
    after=log_retry
)
def call_llm_extract_keywords(chunk_text):
    """
    Chama a LLM para extrair keywords do trecho de texto.
    
    Retorna:
        Objeto KeywordsOutput com a lista de palavras-chave.
    """
    llm = ChatGoogleGenerativeAI(
        api_key=os.getenv("GEMINI_API_KEY"),
        model="models/gemini-1.5-flash",
        temperature=0.7
    )
    prompt = f"""
Aqui está o trecho para geração de keywords:

<chunk> 
{chunk_text}
</chunk>

Gere uma lista de palavras-chave que capture a essência do documento.
(Mínimo 1 e máximo 5)

Responda em JSON:
{{
  "palavras_chave": ["...", ...]
}}
"""
    response = llm.invoke(prompt)
    parser = PydanticOutputParser(pydantic_object=KeywordsOutput)
    try:
        return parser.parse(response.text())
    except Exception as e:
        print("Erro no parse:", e)
        raise e


# -----------------------------------------------------------
# 4. Integração com Pinecone (via LangChain)
# -----------------------------------------------------------
MAX_METADATA_SIZE = 40960  # 40 KB, limite do Pinecone

def get_metadata_size(metadata_list):
    """Calcula o tamanho total dos metadados dos documentos em bytes."""
    return sum(len(json.dumps(doc.metadata, ensure_ascii=False).encode('utf-8')) for doc in metadata_list)


def batch_documents(docs, max_size=MAX_METADATA_SIZE):
    """
    Divide os documentos em lotes garantindo que o tamanho dos metadados seja menor ou igual a max_size.
    """
    batches = []
    current_batch = []
    current_size = 0

    for doc in docs:
        doc_size = len(json.dumps(doc.metadata, ensure_ascii=False).encode('utf-8'))
        if current_size + doc_size > max_size:
            batches.append(current_batch)
            current_batch = []
            current_size = 0
        current_batch.append(doc)
        current_size += doc_size

    if current_batch:
        batches.append(current_batch)
    return batches


@retry(
    stop=stop_after_attempt(20),
    wait=wait_exponential(multiplier=1, min=4, max=60),
    retry=retry_if_exception_type(Exception),
    after=log_retry
)
def upsert_batch_into_pinecone(batch, index_name, vector_db: Optional[PineconeVectorStore] = None):
    """Envia um lote de documentos para o Pinecone."""
    print(f"Enviando lote de {len(batch)} documentos para o Pinecone (Tamanho total: {get_metadata_size(batch)} bytes)")
    if vector_db is None:
        return PineconeVectorStore.from_documents(batch, embeddings, index_name=index_name)
    else:
        return vector_db.from_documents(batch, embeddings, index_name=index_name)


def upsert_chunks_into_pinecone(chunks_with_metadata, index_name):
    """
    Insere ou atualiza os chunks processados no Pinecone.
    
    Retorna True se a operação foi bem-sucedida.
    """
    docs = []
    for item in chunks_with_metadata:
        metadata = dict(item)
        text = metadata.pop("text", "")
        doc = Document(page_content=text, metadata=metadata, id=str(uuid4()))
        docs.append(doc)

    batches = batch_documents(docs)
    processed_docs = []
    vector_db: PineconeVectorStore = None
    for batch in batches:
        try:
            vector_db = upsert_batch_into_pinecone(batch=batch, index_name=index_name)
            processed_docs.extend(batch)
        except Exception as e:
            print(f"Erro ao enviar lote para o Pinecone: {e}")
            print("Deletando documentos já enviados.")
            ids_to_delete = [doc.id for doc in processed_docs]
            if vector_db is not None and ids_to_delete:
                try:
                    vector_db.delete(ids_to_delete)
                except Exception as del_err:
                    print(f"Erro ao deletar documentos: {del_err}")
            return False
    return True


def process_chunk(d, pdf_id, general_metadata: GeneralMetadata):
    """
    Processa cada chunk:
      - Extrai keywords via LLM;
      - Combina informações com metadados gerais e link para o PDF (se disponível).
    """
    chunk_text = d.page_content
    keywords_data = call_llm_extract_keywords(chunk_text)
    result = {
        "text": chunk_text,
        "url": f"https://drive.google.com/file/d/{pdf_id}/view?usp=sharing" if pdf_id else "",
        "titulo_obra": general_metadata.titulo_obra,
        "palavras_chave": keywords_data.palavras_chave,
    }
    if general_metadata.autor:
        result["autor"] = general_metadata.autor
    if general_metadata.bioma:
        result["bioma"] = general_metadata.bioma
    return result


# -----------------------------------------------------------
# 5. Orquestração principal
# -----------------------------------------------------------
def main():
    # Conexão com banco de dados
    conn = get_db_connection('./process_chunks/db/recursive.db')
    create_table_if_not_exists(conn)
    processed_ids = get_processed_file_ids(conn)

    # Inicializa serviço do Google Drive e define a pasta alvo
    drive_service = get_drive_service()
    folder_id = os.getenv("DRIVE_FOLDER_ID", "SEU_FOLDER_ID_AQUI")

    # Lista arquivos Markdown e filtra os que ainda não foram processados
    md_files = list_markdown_files(drive_service, folder_id)
    to_process = [f for f in md_files if f['id'] not in processed_ids]

    # Loop de processamento para cada arquivo novo
    for fobj in to_process:
        file_id = fobj['id']
        file_name = fobj['name']
        print(f"Processando arquivo: {file_name} (ID={file_id})")

        try:
            raw_text = download_file_as_text(drive_service, file_id)
        except Exception as e:
            print(f"Erro ao baixar arquivo {file_id}: {e}")
            continue

        base_name = file_name.rsplit(".", 1)[0]
        pdf_info = find_pdf_with_same_name(drive_service, folder_id, base_name)
        pdf_id = pdf_info["id"] if pdf_info else None
        if pdf_info:
            print(f"Encontrado PDF: {pdf_info['name']} (ID={pdf_id})")
        else:
            print("Nenhum PDF correspondente encontrado.")

        processed_text = extract_text_from_file(raw_text)
        if not processed_text.split():
            print("Arquivo vazio. Pulando.")
            continue

        general_metadata = call_llm_extract_general_metadata(processed_text)
        if general_metadata == {}:
            print("Erro na extração dos metadados gerais. Pulando.")
            continue

        docs = split_text_into_chunks(processed_text, chunk_size=512, chunk_overlap=100)
        chunks_with_metadata = []
        total_chunks = len(docs)
        print(f"{total_chunks} chunks gerados.")
        batch_size = 200

        # Processamento concorrente dos chunks
        for i in range(0, total_chunks, batch_size):
            print(f"Processando lote {i+1} a {i+batch_size}...")
            batch = docs[i:i + batch_size]
            with ThreadPoolExecutor(max_workers=batch_size) as executor:
                future_to_chunk = {executor.submit(process_chunk, d, pdf_id, general_metadata): d for d in batch}
                for future in as_completed(future_to_chunk):
                    try:
                        chunk_result = future.result()
                        chunks_with_metadata.append(chunk_result)
                    except Exception as exc:
                        print("Exceção ao processar um chunk:", exc)

        success = upsert_chunks_into_pinecone(chunks_with_metadata, index_name="seu-index-no-pinecone")
        if success:
            mark_file_as_processed(conn, file_id, file_name)
            print(f"Arquivo {file_name} (ID={file_id}) processado com sucesso.\n")

    conn.close()


if __name__ == "__main__":
    main()
