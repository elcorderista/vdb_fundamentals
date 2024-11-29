from chromadb.utils import embedding_functions
import numpy as np
#Define the default functions for make embeddings, we can use differents providers
default_ef = embedding_functions.DefaultEmbeddingFunction()

name = 'Paulo'

emb = default_ef(name)
emb_array = np.array(emb)

print(emb)
print(emb_array.shape)