# chat_consola.py (Anteriormente main.py)

from dotenv import load_dotenv

# Importaciones existentes
from src.models.ollama_manager import OllamaModelManager
from src.models.chatbot import RAGChatbot
from src.utils.app_utils import load_vector_store
from src.models.traductor import Traductor             # Importa el modelo de traducción
from src.core.rag_service import RAGMultilingualService # Importa el servicio central de lógica

# --- Función Auxiliar para LLM (se mantiene) ---
def select_llm_model(model_manager: OllamaModelManager) -> str:
    """
    Permite al usuario seleccionar un modelo de lenguaje de la lista de disponibles.
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

# --- NUEVA Función Auxiliar para Idioma ---
def select_language() -> str:
    """Permite al usuario seleccionar el idioma de la conversación."""
    print("\nSelecciona el idioma de la conversación:")
    print("1. Español (es)")
    print("2. Gallego (gl)")
    
    while True:
        choice = input("Elige 1 o 2: ")
        if choice == '1':
            return 'es'
        elif choice == '2':
            return 'gl'
        else:
            print("Opción inválida.")
        
# --- Bucle de Chat MODIFICADO ---
# Ahora recibe el servicio central y el idioma
def chat_loop(rag_service: RAGMultilingualService, idioma_conversacion: str):
    """
    Bucle principal de interacción con el chatbot multilingüe.
    """
    print(f" ¡DoBot listo para conversar en {idioma_conversacion.upper()}! Escribe 'salir' para terminar.")
    while True:
        # Pide la entrada en el idioma seleccionado
        user_question = input(f"Tú ({idioma_conversacion.upper()}): ") 
        
        if user_question.lower() == "salir":
            print(" Adiós.")
            break
        
        print(" DoBot: Procesando...")
        try:
            # LLAMADA AL SERVICIO CENTRAL
            response = rag_service.answer_question(
                user_question, 
                idioma_conversacion
            )
            
            # Imprimir la respuesta
            print("-" * 50)
            print("Respuesta:")
            print(response["answer"]) # La respuesta ya viene traducida si aplica
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

# --- Función Principal MODIFICADA ---
def main():
    """
    Función principal que orquesta la ejecución, inicializando todos los componentes.
    """
    # 0. Cargar variables de entorno y seleccionar idioma
    load_dotenv()
    idioma_conversacion = select_language() 

    # 1. Inicializar el gestor de modelos de Ollama
    model_manager = OllamaModelManager()
    
    # 2. Seleccionar el modelo de lenguaje e inicializar LLM
    llm_model_name = select_llm_model(model_manager)
    print(f" Usando el modelo de lenguaje: {llm_model_name}")
    llm = model_manager.create_llm(llm_model_name)
    
    # 3. Cargar la base de datos vectorial
    vector_store, _ = load_vector_store()
    if not vector_store:
        print("No se pudo cargar la base de datos vectorial. Saliendo...")
        return

    # 4. Inicializar RAG Chatbot y Traductor
    chatbot = RAGChatbot(vector_store, llm)
    traductor = Traductor() 

    # 5. Inicializar el Servicio Central de Lógica
    # Este servicio une el chatbot RAG y el traductor
    rag_service = RAGMultilingualService(chatbot, traductor)
    
    # 6. Iniciar el bucle de conversación, pasando el servicio y el idioma
    chat_loop(rag_service, idioma_conversacion)

if __name__ == "__main__":
    main()