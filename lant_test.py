# =============================
# === 1 Library imports
# =============================
import os
from dotenv import load_dotenv
from langchain_core.messages  import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

# =============================
# === 2 Global variables
# =============================
load_dotenv()
openai_api_key = os.getenv('OPENAI_API_KEY')
MODEL_NAME = 'gpt-4o-mini'

# =============================
# === 3 Declaration Model
# =============================

model = ChatOpenAI(api_key=openai_api_key, model_name=MODEL_NAME)


# =============================
# === 4 Generacion del Mensaje
# =============================
messages = [
    SystemMessage("Translate the following from English into Russian"),
    HumanMessage("hi!"),
]

# =============================
# === 5 Invocacion del Modelo
# =============================
response = model.invoke(messages)
print(response)
print(response.content)