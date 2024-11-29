import os
from dotenv import load_dotenv
from openai import OpenAI

#Load Env variables
load_dotenv()
client_openAi = OpenAI(api_key=os.getenv("OPEN_API_KEY"))

#Requerst embeddings from text supply
response = client_openAi.embeddings.create(
    input='Your text string goes here', model="text-embedding-3-small"
)

print(response)