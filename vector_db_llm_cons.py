# =============================
# === 1 Importacion de librerias
# =============================
import chromadb
from openai import OpenAI
from chromadb.utils import embedding_functions
import os
from dotenv import load_dotenv
# =============================
# === 2 Load API KEY
# =============================
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
client_openAI = OpenAI(api_key=openai_api_key)
PATH_PERSISTANCE = './db/chroma_persist_storage'
CHAT_MODEL = "gpt-3.5-turbo"

# =============================
# === 3 Define Embedding Functions
# =============================

openai_ef = embedding_functions.OpenAIEmbeddingFunction(
    api_key=openai_api_key,
    model_name='text-embedding-3-small'
)

# =============================
# === 4 Get Persistances Embeddings
# =============================
chorma_cliente = chromadb.PersistentClient(
    path='./db/chroma_persist_storage'
)
collection_name = 'document_qa_collection'
collection = chorma_cliente.get_collection(
    name=collection_name,
    embedding_function=openai_ef
)

# =============================
# === 5 Get Relevant chunks
# =============================
def query_documents(question, n_results=2):
    results = collection.query(
        query_texts=question,
        n_results=n_results
    )

    #Extract relevant chunks
    relevant_chunks = [doc for sublist in results["documents"] for doc in sublist]
    print("=== Returining relevant chunks ===")
    return relevant_chunks

# =============================
# === 6 Generate response from P
# =============================
def generate_response(question, relevant_chunks):
    context = '\n\n'.join(relevant_chunks)
    prompt = (
        "You are an assistant for question-answering tasks. Use the following pieces of "
        "retrieved context to answer the question. If you don't know the answer, say that you "
        "don't know. Use three sentences maximum and keep the answer concise."
        "\n\nContext:\n" + context + "\n\nQuestion:\n" + question
    )

    response = client_openAI.chat.completions.create(
        model= CHAT_MODEL,
        messages=[
            {
                'role': "system",
                'content': prompt,
            },
            {
                'role': "user",
                'content': question,
            },
        ],
    )

    answer = response.choices[0].message
    return answer


question = "Cuentame sobre la adquisicion de Okera"
relevant_chunks = query_documents(question)
answer = generate_response(question, relevant_chunks)

print("=== Relevant chunks ===")
for chunk in relevant_chunks:
    print("===")
    print(chunk)

print("=== Answer ===")
print(answer.content)
