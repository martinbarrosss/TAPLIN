# sim_test.py

import argparse
import sys
from src.pipeline.batch_tester import run_evaluation_batch

# --- CONFIGURACIÓN DE LA PRUEBA ---
THEME="Frigo"
MODEL="Kimi"
# ------------------
INPUT_FILE_NAME = f"Prompts{THEME}.csv"
OUTPUT_FILE_NAME = f"{THEME}Results{MODEL}.csv"
# ------------------
DEFAULT_JUDGE = "gpt-oss:20b-cloud"  # El modelo juez por defecto

# --- LISTA DE MODELOS OFICIALES ---
AVAILABLE_MODELS = [
    "gpt-oss:20b-cloud",
    "deepseek-v3.1:671b-cloud",
    "kimi-k2:1t-cloud",
    "qwen3-coder:480b-cloud"
]

def select_model_interactively():
    """Muestra un menú para elegir el modelo si no se pasó por argumento."""
    print("\n--- SELECCIÓN DE MODELO PARA SIMULACIÓN ---")
    for i, model_name in enumerate(AVAILABLE_MODELS):
        print(f"  [{i+1}] {model_name}")
    
    while True:
        try:
            choice_raw = input(f"\nSelecciona el número del modelo (1-{len(AVAILABLE_MODELS)}): ")
            choice = int(choice_raw) - 1
            if 0 <= choice < len(AVAILABLE_MODELS):
                return AVAILABLE_MODELS[choice]
            else:
                print(" Opción no válida. Intenta de nuevo.")
        except ValueError:
            print(" Por favor, introduce un número válido.")

def run_simulation(target_model: str = None):
    """
    Función principal del simulador.
    """
    print("==============================================")
    print("   INICIO DEL SIMULADOR DE PRUEBAS MASIVAS    ")
    print("==============================================")
    
    selected_model = target_model

    # Si no se especificó modelo por argumentos, preguntamos al usuario
    if not selected_model:
        selected_model = select_model_interactively()
    else:
        # Validamos si el modelo pasado por argumento es conocido
        if selected_model not in AVAILABLE_MODELS:
            print(f"  Aviso: El modelo '{selected_model}' no está en la lista oficial, pero se intentará usar.")

    print(f"\n Modelo seleccionado: {selected_model}")
    print(f"  Juez asignado: {DEFAULT_JUDGE}")

    run_evaluation_batch(
        input_filename=INPUT_FILE_NAME,
        output_filename=OUTPUT_FILE_NAME,
        models_to_test=[selected_model], 
        judge_model=DEFAULT_JUDGE
    )

    print("\n============================================")
    print("   FIN DEL SIMULADOR DE PRUEBAS MASIVAS       ")
    print("==============================================")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Ejecuta pruebas masivas contra modelos de lenguaje."
    )
    parser.add_argument(
        "--model",
        default=None,
        help="(Opcional) Nombre del modelo. Si se omite, se abrirá un menú interactivo."
    )
    
    args = parser.parse_args()
    
    try:
        run_simulation(target_model=args.model)
    except KeyboardInterrupt:
        print("\n\n Ejecución interrumpida por el usuario.")
        sys.exit(0)