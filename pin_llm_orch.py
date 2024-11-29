# ===============================
# Libraries
# ===============================
import os

from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from pinecone import Pinecone, ServerlessSpec

from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain

from langchain_pinecone import PineconeVectorStore

# ===============================
# Load Env Variables
# ===============================
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
EMBEDDINGS_MODEL = os.getenv("EMBEDDINGS_MODEL")
CHAT_MODEL = os.getenv("CHAT_MODEL")
DOCUMENTS_PATH = os.getenv("DOCUMENTS_PATH")

# ===============================
# Load Documents
# ===============================
loader = DirectoryLoader(
    path=DOCUMENTS_PATH,
    glob='*.txt',
    loader_cls=TextLoader
)
print('=== Load Documents ===')
documents = loader.load()
print(f'=== Total Load Documents: {len(documents)}')

# ===============================
# Create Chuncks
# ===============================
text_splitter = RecursiveCharacterTextSplitter(
    separators=["\n\n", "\n"],
    chunk_size=1000,
    chunk_overlap=20,
)
print("=== Text Splitter ===")

chunks = text_splitter.split_documents(documents)
print(f"=== Total chinks: {len(chunks)}")
# ===============================
# Create Embeddings and ChatModel
# ===============================
embeddings = OpenAIEmbeddings(
    api_key=OPENAI_API_KEY,
    model=EMBEDDINGS_MODEL,
)
print("=== OpenAI Embeddings ===")

llm = ChatOpenAI(
    api_key=OPENAI_API_KEY,
    model=CHAT_MODEL,
)
print("=== ChatOpenAI ===")
# ===============================
# Create Persistence
# ===============================
pc = Pinecone(api_key=PINECONE_API_KEY)


index_name = 'tester-index'
existing_indexes = [index_info['name'] for index_info in pc.list_indexes()]

if index_name not in existing_indexes:
    print("=== Creating PineCone Index ===")
    pc.create_index(
        name=index_name,
        dimension=3072,
        metric='cosine',
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1"
        ),
    )
index = pc.Index(index_name)



if index_name not in existing_indexes:
    print("Creando el índice y cargando los datos...")
    vectorstore_from_docs = PineconeVectorStore.from_documents(
        index_name=index_name,
        embedding=embeddings,
        documents=chunks)

else:
    print(f"Usando el índice existente: {index_name}")
    vectorstore_from_docs = PineconeVectorStore(
        index_name=index_name,
        embedding=embeddings
    )


retriever = vectorstore_from_docs.as_retriever()
print("=== Retriever ===")
# ===============================
# Handle Queries
# ===============================
#Define prompt
system_prompt = (
    "You are an assistant for question-answering tasks. "
    "Use the following pieces of retrieved context to answer "
    "the question. If you don't know the answer, say that you "
    "don't know. Use three sentences maximum and keep the "
    "answer concise."
    "\n\n"
    "{context}"
)
#Define roles and dinamics vairables
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ('human', "{input}")
    ]
)
#Make unify output with cahin answer-question
question_answer_chain = create_stuff_documents_chain(
    llm=llm,
    prompt=prompt,
)
#Make Chain Retrieval - Ansewer
rag_chain = create_retrieval_chain(retriever, question_answer_chain)

response = rag_chain.invoke({"input": "Que nuevas noticias hay de databricks"})
res = response["answer"]

print(res)
