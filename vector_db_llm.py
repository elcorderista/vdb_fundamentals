# =============================
# === 1 Importacion de librerias
# =============================
import os
from dotenv import load_dotenv
import chromadb
from chromadb.utils import embedding_functions
from openai import OpenAI

# =============================
# === 2 Load API KEY
# =============================
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
client_openAI = OpenAI(api_key=openai_api_key)

# =============================
# === 3 Define Embedding Functions
# =============================
default_ef = embedding_functions.DefaultEmbeddingFunction()
openai_ef = embedding_functions.OpenAIEmbeddingFunction(
    api_key=openai_api_key,
    model_name='text-embedding-3-small'
)
# =============================
# === 4 Create Persistances Embeddings
# =============================
chorma_cliente = chromadb.PersistZentClient(
    path='./db/chroma_persist_storage'
)
collection_name = 'document_qa_collection'
collectoin = chorma_cliente.get_or_create_collection(
    name=collection_name,
    embedding_function=openai_ef
)

# =============================
# === 5 Load Documentos from a directory
# =============================
def load_documents_from_directory(directory_path):
    print("==== Loading documents from directory ====")
    documents = []
    for filename in os.listdir(directory_path):
        if filename.endswith(".txt"):
            with open(
                    os.path.join(directory_path, filename), "r", encoding="utf-8"
            ) as file:
                documents.append({"id": filename, "text": file.read()})
    return documents

#Load documents from the directory
directory_path = './data/new_articles'
documents = load_documents_from_directory(directory_path)
print(f'Loaded {len(documents)} documents')


# =============================
# === 6 Split Documentos in Chunks
# =============================
def split_text(text, chunk_size=1000, chunk_overlap=20):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - chunk_overlap
    return chunks

chunked_documents = []
for document in documents:
    chunks = split_text(document['text'])
    print('==== Splitting docs into chunks ====')
    for i, chunk in enumerate(chunks):
        chunked_documents.append({"id": f"{document['id']}_chunk{i+1}", "text": chunk})

# =============================
# === 7 Generate embeddings
# =============================
def get_openai_embeddings(text):
    response = client_openAI.embeddings.create(
        input=text,
        model='text-embedding-3-small'
    )
    embedding = response.data[0].embedding
    print('==== Generating embeddings... ====')
    return embedding

for doc in chunked_documents:
    print("==== Generating embeddings... ====")
    doc['embedding'] = get_openai_embeddings(doc['text'])

# =============================
# === 8 Upsert documents with embeddings into chromadb
# =============================
for doc in chunked_documents:
    print("==== Inserting chunks into db;;; ====")
    collectoin.upsert(
        ids=[doc['id']], documents=[doc['text']], embeddings=[doc['embedding']]
    )

