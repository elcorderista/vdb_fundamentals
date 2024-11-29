# =============================
# === 1 Library imports
# =============================
import os
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain

# =============================
# === 2 Global variables
# =============================
load_dotenv()
openai_api_key = os.getenv('OPENAI_API_KEY')
MODEL_NAME = 'gpt-4o-mini'
MODEL_EMBEDDING = 'text-embedding-3-small'
PERSIS_DIRECTORY = './db/chroma_db_real_world'

# =============================
# === 3 Load Documents
# =============================
loader = DirectoryLoader(
    path='./data/new_articles',
    glob="*.txt",
    show_progress=True,
    use_multithreading=True,
    loader_cls=TextLoader
)
documents = loader.load()
print('=== Load Documents ===')
print(f'Total documents: {len(documents)}')

# =============================
# === 4 Split Texts
# =============================
text_splitter = RecursiveCharacterTextSplitter(
    separators=["\n\n", "\n"],
    chunk_size=1000,
    chunk_overlap=20,
)
print('=== Split Documents ===')
spliteded_documents = text_splitter.split_documents(documents)
print(f'Total documents: {len(spliteded_documents)}')

# =============================
# === 5 Create Embeddings and Vector Database
# =============================
embeddings = OpenAIEmbeddings(
    api_key=openai_api_key,
    model=MODEL_EMBEDDING,
)
persist_derectory = PERSIS_DIRECTORY
vector_db = Chroma.from_documents(
    documents=spliteded_documents,
    embedding=embeddings,
    persist_directory=persist_derectory
)
retriever = vector_db.as_retriever()
#res_docs = retriever.invoke("how much did microft raise?", k=2)
#print(res_docs)
