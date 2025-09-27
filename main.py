# Main temporal chat, para runear desde aquí poco a poco el proceso que vayamos haciendo

from Indexing.preprocesamiento import cargar_y_fragmentar
from Indexing.embeddings import crear_base_vectorial
from Rag.generador import inicializar_qa_chain # Importar la nueva función
import os

# --- CONFIGURACIÓN GLOBAL ---
PDF_FILE_PATH = "Files/InfoNevera.pdf"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 100 

EMBEDDING_MODEL_NAME = "BAAI/bge-m3" 
CHROMA_PERSIST_DIR = "DataVec_Nevera"
LLM_MODEL_NAME = "meta-llama/Llama-2-7b-chat-hf" 


def chat_loop(qa_chain):
    """Bucle de chat interactivo con el usuario."""
    print("🤖 NeveraBot: ¡Hola! Pregúntame sobre el manual de tu frigorífico/congelador.")
    print("Escriba 'salir' o 'exit' para terminar la conversación.")
    
    while True:
        try:
            # Cogemos la pregunta de ramets
            question = input("👤 Usuario: ")
            
            if question.lower() in ["salir", "exit", "quit"]:
                print("🤖 NeveraBot: ¡Adiós! Ha sido un placer ayudarle con su nevera.")
                break
            
            if not question.strip():
                continue

            # Ejecuta la cadena RAG
            print("\n🤖 NeveraBot (Buscando contexto y generando respuesta...)")
            resultado = qa_chain.invoke({"query": question})
            
            # Mostramos la respuesta y las fuentes
            print("\n" + "="*50)
            print("🤖 NeveraBot: " + resultado['result'].strip())
            print("="*50)

        except Exception as e:
            print(f"Ocurrió un error en el chat: {e}")
            print("Intente de nuevo o escriba 'salir'.")

if __name__ == "__main__":

    # INDEXACIÓN (Carga y Fragmentación)
    if not os.path.exists(PDF_FILE_PATH):
        print(f"¡ERROR! No se encontró el archivo: {PDF_FILE_PATH}")
        print("Asegúrese de que el archivo esté en 'Files/InfoNevera.pdf'.")
        exit()
        
    print("--- FASE DE INDEXACIÓN ---")
    chunks = cargar_y_fragmentar(PDF_FILE_PATH, CHUNK_SIZE, CHUNK_OVERLAP)
    
    # VECTORIZACIÓN Y ALMACENAMIENTO (Crea el índice si no existe)
    vector_store, embeddings = crear_base_vectorial(chunks, EMBEDDING_MODEL_NAME, CHROMA_PERSIST_DIR)

    # INICIALIZACIÓN DEL RAG (Carga el LLM y la cadena QA)
    if vector_store:
        qa_chain = inicializar_qa_chain(vector_store, LLM_MODEL_NAME)
        
        # INICIO DEL CHAT INTERACTIVO
        chat_loop(qa_chain)