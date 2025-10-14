import os
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

# Asume que la base de datos está en la raíz del proyecto.
PERSIST_DIR = "./chroma_db" 

def load_vector_store(persist_dir: str = PERSIST_DIR):
    """Carga la base de datos vectorial persistente."""
    if not os.path.exists(persist_dir):
        return None, PERSIST_DIR
    
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    vector_store = Chroma(
        persist_directory=persist_dir,
        embedding_function=embeddings,
        collection_name="manuales_electrodomesticos"
    )
    return vector_store, PERSIST_DIR
