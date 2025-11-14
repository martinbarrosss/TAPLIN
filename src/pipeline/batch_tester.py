# src/pipeline/batch_tester.py 

import os
import pandas as pd
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma 

# Nota: Imports absolutos para asegurar la compatibilidad cuando se ejecute desde 'sim_test.py'
from src.models.chatbot import RAGChatbot
from src.models.traductor import Traductor
from src.models.ollama_manager import OllamaModelManager
from src.core.rag_service import RAGMultilingualService
from src.utils.app_utils import load_vector_store 

# --- Rutas Base (Carpeta donde se encuentran los archivos) ---
# Usamos os.path.join para construir rutas robustas
# Nota: Asumo que la carpeta padre de input/output es 'testing' o 'tests' en la raíz del proyecto.
INPUT_DIR = "testing/input"
OUTPUT_DIR = "testing/output"

# Variables globales que se llenarán con las rutas completas en la función principal
GLOBAL_INPUT_PATH = None
GLOBAL_OUTPUT_PATH = None

# --- Funciones Auxiliares ---

def initialize_static_components(ollama_manager: OllamaModelManager) -> tuple[Traductor, Chroma]:
    """
    Inicializa los componentes que son estáticos y no cambian entre modelos (Traductor y Vector Store).
    
    Returns: Una tupla con (Traductor, Vector Store)
    """
    print("🛠️ Inicializando componentes estáticos (Traductor y Chroma)...")
    try:
        traductor = Traductor()
        vector_store, _ = load_vector_store()
        
        if vector_store is None:
            raise FileNotFoundError("No se pudo cargar la base de datos Chroma.")
            
        print("✅ Componentes estáticos inicializados.")
        return traductor, vector_store
    
    except Exception as e:
        print(f"🛑 Error crítico en la inicialización: {e}")
        raise

def load_input_queries() -> pd.DataFrame:
    """
    Carga y valida el archivo CSV de consultas usando la ruta global.
    
    Returns: DataFrame de Pandas con las consultas.
    """
    print(f"📄 Cargando consultas desde '{GLOBAL_INPUT_PATH}'...")
    try:
        # Nota: pd.read_csv usa la coma (,) como separador por defecto, lo cual es correcto para el input.
        df_input = pd.read_csv(GLOBAL_INPUT_PATH)
        
        if df_input.empty or 'query' not in df_input.columns:
            raise ValueError(f"El archivo '{GLOBAL_INPUT_PATH}' está vacío o no tiene la columna 'query'.")
        
        # Asumiendo que DEFAULT_LANG está definido o se usa el valor literal 'es'
        DEFAULT_LANG = "es" 
        if 'language' not in df_input.columns:
            df_input['language'] = DEFAULT_LANG
            
        print(f"✅ {len(df_input)} consultas cargadas correctamente.")
        return df_input
        
    except FileNotFoundError:
        print(f"🛑 Error: Archivo de entrada '{GLOBAL_INPUT_PATH}' no encontrado.")
        raise
    except ValueError as e:
        print(f"🛑 Error de formato: {e}")
        raise

def process_model_queries(model_name: str, df_input: pd.DataFrame, traductor: Traductor, vector_store: Chroma, ollama_manager: OllamaModelManager) -> list[str]:
    """
    Ejecuta todas las consultas para un modelo específico.
    """
    TEMPERATURE = 0.2
    
    print(f"\n--- 🤖 PROBANDO MODELO: {model_name} ---")
    results = []
    
    try:
        # 1. Crear Instancia LLM
        llm_instance = ollama_manager.create_llm(model_name, TEMPERATURE)
        
        # 2. Inicializar la cadena RAG y el Servicio
        rag_chatbot = RAGChatbot(vector_store=vector_store, llm=llm_instance)
        rag_service = RAGMultilingualService(chatbot=rag_chatbot, traductor=traductor)
        
        # 3. Iterar sobre las Consultas
        for index, row in df_input.iterrows():
            user_query = row['query']
            lang = row['language']
            
            print(f"  [{index+1}/{len(df_input)}] Q({lang}): {user_query[:50]}...")
            
            # 4. Ejecutar la Consulta RAG Multilingüe
            response_data = rag_service.answer_question(
                user_prompt=user_query, 
                idioma_conversacion=lang
            )
            
            # Guardamos la respuesta final
            results.append(response_data['answer'])
            
    except Exception as e:
        error_message = f"ERROR_MODELO_CRITICO: {str(e)[:50]}..."
        print(f"  ❌ Error crítico al procesar el modelo {model_name}: {error_message}")
        
        # Llenar las respuestas restantes con el mensaje de error
        results.extend([error_message] * (len(df_input) - len(results)))
    
    return results


def run_batch_test_from_script(input_filename: str, output_filename: str):
    """
    Función principal ORQUESTADORA, ahora toma los nombres de archivo como parámetros.
    """
    global GLOBAL_INPUT_PATH
    global GLOBAL_OUTPUT_PATH
    
    # CORRECCIÓN DE RUTA: Uso de os.path.join para compatibilidad
    GLOBAL_INPUT_PATH = os.path.join(INPUT_DIR, input_filename)
    GLOBAL_OUTPUT_PATH = os.path.join(OUTPUT_DIR, output_filename)
    
    # Asegurar que el directorio de salida exista
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print("🚀 Iniciando el Orquestador de Pruebas Masivas...")
    load_dotenv()
    ollama_manager = OllamaModelManager()
    
    try:
        # 1. Inicialización de componentes estáticos
        traductor, vector_store = initialize_static_components(ollama_manager)
        
        # 2. Carga de consultas de entrada
        df_input = load_input_queries()
        
    except Exception:
        # Si falla initialize_static_components o load_input_queries, el script termina.
        return
    
    df_output = df_input.copy()
    available_models = ollama_manager.get_available_models()
    
    # 3. Procesamiento por cada modelo
    print(f"\nTotal de modelos a probar: {len(available_models)}")

    for model_name in available_models:
        
        results = process_model_queries(
            model_name=model_name,
            df_input=df_input,
            traductor=traductor,
            vector_store=vector_store,
            ollama_manager=ollama_manager
        )
        
        # 4. Agregar resultados al DataFrame de salida
        df_output[model_name] = results

    # 5. Escribir el CSV de Salida
    print(f"\n✅ Pruebas finalizadas. Guardando resultados en '{GLOBAL_OUTPUT_PATH}'...")
    
    # *** CORRECCIÓN CRÍTICA: Usamos punto y coma (;) como separador para evitar la corrupción del CSV ***
    df_output.to_csv(GLOBAL_OUTPUT_PATH, index=False, sep=';')
    
    print("¡Proceso completado!")


if __name__ == "__main__":
    # Si se ejecuta directamente (ejecución por defecto), usa un nombre por defecto
    run_batch_test_from_script("input_queries.csv", "output_results.csv")