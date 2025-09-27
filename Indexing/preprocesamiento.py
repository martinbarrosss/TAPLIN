
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

def cargar_y_fragmentar(pdf_path: str, chunk_size: int, chunk_overlap: int):

    print(f"--- 1. Cargando el documento: {pdf_path} ---")

    loader = PyPDFLoader(pdf_path)
    documentos = loader.load()

    print(f"Documento cargado con {len(documentos)} páginas.")
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""]
    )

    print(f"--- 2. Fragmentando el texto (Chunk Size: {chunk_size}, Overlap: {chunk_overlap}) ---")

    fragmentos = text_splitter.split_documents(documentos)

    print(f"Proceso finalizado. Se han generado {len(fragmentos)} fragmentos (chunks).")
    print("-" * 30)
    print("Ejemplo del primer fragmento (metadatos y contenido):")
    print(f"Fuente (página): {fragmentos[0].metadata['page']}")
    print(f"Contenido:\n{fragmentos[0].page_content[:500]}...") # Mostrar solo los primeros 500 caracteres

    return fragmentos