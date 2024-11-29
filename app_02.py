import chromadb
from chromadb.utils import embedding_functions

#Create the client
chromadb_client = chromadb.Client()
default_ef = embedding_functions.DefaultEmbeddingFunction()

#Named the collection
collection_name = 'test_collection'

#Handle the collection
collection = chromadb_client.get_or_create_collection(collection_name, embedding_function=default_ef)

#Define text documents
documents = [
    {"id": "doc1", "text": "Hello, world!"},
    {"id": "doc2", "text": "How are you today?"},
    {"id": "doc3", "text": "Goodbye, see you later!"},
]

#Add Documents
for doc in documents:
    collection.upsert(ids=doc["id"], documents=[doc["text"]])


#Define query text
query_text = 'Hello, world!'

#Define the schema result
results = collection.query(
    query_texts=[query_text],
    n_results=3,
)

#Review Results
for idx, document in enumerate(results['documents'][0]):
    doc_id = results['ids'][0][idx]
    distance = results['distances'][0][idx]
    print(
        f'For the query: {query_text}, \n Found similar documents {document}, ID:{doc_id} \n Distance: {distance}'
    )

