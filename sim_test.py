# sim_test.py

# Importa la función de orquestación desde tu script
from src.pipeline.batch_tester import run_batch_test_from_script


# --- CONFIGURACIÓN DE LA PRUEBA ---
INPUT_FILE_NAME = "PromptsFrigo.csv"
OUTPUT_FILE_NAME = "FrigoResults.csv"

def run_simulation():
    """
    Función que simula el punto de entrada para las pruebas.
    """
    print("==============================================")
    print("   INICIO DEL SIMULADOR DE PRUEBAS MASIVAS    ")
    print("==============================================")
    
    # Llama a la función principal del batch_tester, pasándole los nombres de archivo
    run_batch_test_from_script(
        input_filename=INPUT_FILE_NAME,
        output_filename=OUTPUT_FILE_NAME
    )
    
    print("\n==============================================")
    print("   FIN DEL SIMULADOR DE PRUEBAS MASIVAS       ")
    print("==============================================")


if __name__ == "__main__":
    run_simulation()