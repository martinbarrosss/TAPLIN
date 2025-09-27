import os
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.vectorstores import Chroma

def crear_base_vectorial(fragmentos, EMBEDDING_MODEL_NAME, CHROMA_PERSIST_DIR):

    print(f"--- 2. Cargando modelo de Embedding: {EMBEDDING_MODEL_NAME} ---")

    # Inicializamos el modelo de Hugging Face.
    embeddings = HuggingFaceBgeEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={'device': 'cpu'}, # Use 'cuda' si tiene una GPU NVIDIA
        encode_kwargs={'normalize_embeddings': True}
    )

    print(f"--- 3. Creando Base de Datos Vectorial (ChromaDB) y guardando los Embeddings ---")

    # Creamos la base de datos vectorial a partir de los fragmentos y el modelo de embeddings.
    vector_store = Chroma.from_documents(
        documents=fragmentos,
        embedding=embeddings,
        persist_directory=CHROMA_PERSIST_DIR        # Este parámetros creo que nos asegura que se fuarde en disco
    )
    
    # Guardar en disco (Chroma lo hace automáticamente con .from_documents, pero lo confirmamos)
    vector_store.persist()
    
    print("-" * 50)
    print(f"¡Base de datos vectorial creada con éxito!")
    print(f"Los datos se han guardado en la carpeta: {CHROMA_PERSIST_DIR}")
    print("Su sistema RAG está listo para la fase de consulta.")
    
    return vector_store, embeddings