# main.py

from models.ollama_manager import OllamaModelManager
from models.chatbot import RAGChatbot
from dotenv import load_dotenv
from utils.app_utils import load_vector_store

# --- Configuración de constantes ---
VECTOR_DB_PATH = "../../chroma_db"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2" 

def select_llm_model(model_manager: OllamaModelManager) -> str:
    """
    Permite al usuario seleccionar un modelo de lenguaje de la lista de disponibles.
    
    Returns:
        El nombre del modelo seleccionado por el usuario.
    """
    available_models = model_manager.get_available_models()
    
    print("Modelos de Ollama disponibles:")
    for i, model_name in enumerate(available_models):
        print(f"{i+1}. {model_name}")
        
    while True:
        try:
            choice = int(input("Selecciona un modelo por su número: "))
            if 1 <= choice <= len(available_models):
                llm_model_name = available_models[choice - 1]
                return llm_model_name
            else:
                print("Opción inválida. Por favor, elige un número de la lista.")
        except ValueError:
            print("Entrada inválida. Por favor, introduce un número.")
        
def chat_loop(chatbot: RAGChatbot):
    """
    Bucle principal de interacción con el chatbot.
    """
    print(" ¡DoBot listo! Escribe 'salir' para terminar.")
    while True:
        user_question = input("Tú: ")
        if user_question.lower() == "salir":
            print(" Adiós.")
            break
        
        print(" DoBot: Procesando...")
        try:
            response = chatbot.answer_question(user_question)
            
            # Imprimir la respuesta
            print("-" * 50)
            print("Respuesta:")
            print(response["answer"])
            print("-" * 50)
            
            # Imprimir las fuentes
            print("Fuentes:")
            if response["sources"]:
                for source in sorted(list(set(response["sources"]))):
                    print(f"- {source}")
            else:
                print("- No se encontraron fuentes relevantes.")
            print("-" * 50)
            
        except Exception as e:
            print(f" Ocurrió un error al procesar la pregunta: {e}")

def main():
    """
    Función principal que orquesta la ejecución del chatbot RAG.
    """
    # 0. Cargar variables de entorno
    load_dotenv()

    # 1. Inicializar el gestor de modelos de Ollama
    model_manager = OllamaModelManager()
    
    # 2. Seleccionar el modelo de lenguaje
    llm_model_name = select_llm_model(model_manager)
    print(f" Usando el modelo de lenguaje: {llm_model_name}")
    llm = model_manager.create_llm(llm_model_name)
    
    # 3. Cargar la base de datos vectorial
    vector_store, _ = load_vector_store()
    if not vector_store:
        print("No se pudo cargar la base de datos vectorial. Saliendo...")
        return

    # 4. Inicializar el RAG Chatbot
    chatbot = RAGChatbot(vector_store, llm)
    
    # 5. Iniciar el bucle de conversación
    chat_loop(chatbot)

if __name__ == "__main__":
    main()