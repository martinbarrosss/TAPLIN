# view_results.py

import pandas as pd
import os
import sys

# --- CONFIGURACIÓN ---
CSV_SEPARATOR = ';' 
# Ruta por defecto al archivo de salida
DEFAULT_OUTPUT_PATH = "testing/output/test_results_v1.csv"
MAX_DISPLAY_ROWS = 50 # Límite de filas a mostrar en el menú

def load_data(file_path: str) -> pd.DataFrame | None:
    """Carga el DataFrame desde el archivo CSV de resultados."""
    try:
        # Cargar el DataFrame usando el separador correcto
        df = pd.read_csv(file_path, sep=CSV_SEPARATOR)
        return df
    except FileNotFoundError:
        print(f"\n❌ Error: Archivo de resultados no encontrado en: {file_path}")
        print("Asegúrate de que la ruta es correcta y has ejecutado 'sim_test.py' antes.")
        return None
    except Exception as e:
        print(f"\n❌ Error al leer el CSV. ¿Es correcto el separador ('{CSV_SEPARATOR}')?: {e}")
        return None

def display_question_details(df: pd.DataFrame, question_index: int):
    """Muestra la respuesta de todos los modelos para una única pregunta."""
    
    # Seleccionar la fila de la pregunta
    row = df.iloc[question_index]
    
    # Identificar las columnas de los modelos
    model_columns = [col for col in df.columns if col not in ['query', 'language']]

    print("\n" + "="*80)
    print(f"👀 VISTA DE RESPUESTAS (Pregunta {question_index})")
    print("="*80)
    
    # Mostrar la pregunta original
    print(f"** Pregunta: {row['query']}")
    print(f"** Idioma: {row['language']}")
    print("-" * 80)
    
    # Mostrar las respuestas de los modelos
    for model in model_columns:
        response = row[model].strip()
        print(f"\n🤖 MODELO: {model}")
        print("-" * (len(model) + 10))
        # Reemplazar saltos de línea para una visualización más limpia en la consola
        print(response.replace('\n', ' ').replace('\r', ' '))
    
    print("\n" + "="*80)
    print("FIN DE RESULTADOS")
    print("="*80)

def interactive_menu(df: pd.DataFrame):
    """Muestra el menú interactivo y maneja la entrada del usuario."""
    
    num_questions = len(df)
    # Definir el límite real de visualización (min(total, 50))
    display_limit = min(num_questions, MAX_DISPLAY_ROWS)
    
    while True:
        print("\n" + "#" * 50)
        print(f"MENÚ PRINCIPAL ({num_questions} preguntas totales)")
        print("#" * 50)
        
        # 1. Mostrar las preguntas disponibles
        print(f"Preguntas disponibles (0 a {display_limit - 1}):")
        
        for i in range(display_limit):
            query = df.iloc[i]['query']
            lang = df.iloc[i]['language']
            print(f" [{i:02}] ({lang}) {query[:70]}...")

        if num_questions > MAX_DISPLAY_ROWS:
             print(f"\n[... Se han omitido {num_questions - MAX_DISPLAY_ROWS} preguntas. El límite es {MAX_DISPLAY_ROWS}].")
        
        print("\n---")
        # 2. Pedir entrada al usuario
        user_input = input(f"Introduce el índice [0-{display_limit - 1}] para ver detalles, o 'q' para salir: ").strip()
        print("---")
        
        if user_input.lower() == 'q':
            print("Saliendo del visualizador. ¡Hasta pronto!")
            break
            
        try:
            index = int(user_input)
            
            # 3. Validar el índice
            if 0 <= index < display_limit:
                display_question_details(df, index)
            elif index >= display_limit and index < num_questions:
                 # Si introduce un índice válido pero fuera del límite del menú
                 display_question_details(df, index)
            else:
                print(f"⚠️ Entrada inválida. El índice debe estar entre 0 y {num_questions - 1}.")
                
        except ValueError:
            print("⚠️ Entrada inválida. Por favor, introduce un número o 'q'.")


if __name__ == "__main__":
    # La ruta por defecto puede ser configurada aquí o pasarse como argumento si es necesario
    
    # Para simplicidad y siguiendo la idea de un script ejecutable:
    data_frame = load_data(DEFAULT_OUTPUT_PATH)
    
    if data_frame is not None:
        interactive_menu(data_frame)