from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.core.settings import Settings
from llama_index.embeddings.google_genai import GoogleGenAIEmbedding
from llama_index.core import StorageContext, load_index_from_storage
from llama_index.llms.google_genai import GoogleGenAI


import os

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "AIzaSyDEylsUWBxRbX_6YjyRwZtwxzl7Hj_lH5o")
if GOOGLE_API_KEY:
    os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
else:
    raise ValueError("GOOGLE_API_KEY is not set.")


Settings.llm  = GoogleGenAI(
    model="models/gemini-2.5-flash",
    api_key=GOOGLE_API_KEY,
)

#  Make sure to run pip install llama-index-embeddings-gemini

Settings.embed_model = GoogleGenAIEmbedding(
    model_name="models/gemini-embedding-001",
    api_key=GOOGLE_API_KEY
)

# Create embeddings from scratch
# documents = SimpleDirectoryReader("data").load_data()
# index = VectorStoreIndex.from_documents(documents, show_progress=True)
# index.storage_context.persist()


# rebuild storage context
storage_context = StorageContext.from_defaults(persist_dir="prof_embeddings_demo")

# load index
index = load_index_from_storage(storage_context)


