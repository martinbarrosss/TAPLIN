# evaluator.py

import pandas as pd
import glob
import argparse
import os

def load_and_combine_results(file_paths: list[str]) -> pd.DataFrame | None:
    """
    Carga múltiples archivos CSV de resultados, los combina en un único DataFrame
    y valida que tengan las columnas necesarias.
    """
    all_dfs = []
    required_columns = {'model', 'answer', 'sense_score', 'latency_seconds'}

    print(f"🔎 Encontrados {len(file_paths)} archivos para analizar...")

    for path in file_paths:
        try:
            df = pd.read_csv(path, sep=';')
            if not required_columns.issubset(df.columns):
                print(f" Aviso: El archivo '{path}' no contiene las columnas requeridas ({', '.join(required_columns)}) y será ignorado.")
                continue
            print(f"  - Cargando '{os.path.basename(path)}' ({len(df)} filas)...")
            all_dfs.append(df)
        except FileNotFoundError:
            print(f" Error: Archivo no encontrado en '{path}'.")
        except Exception as e:
            print(f" Error al leer el archivo '{path}': {e}")
    
    if not all_dfs:
        print("\n No se pudo cargar ningún archivo de resultados válido.")
        return None
    
    combined_df = pd.concat(all_dfs, ignore_index=True)
    print(f"\n Total de {len(combined_df)} registros cargados para el análisis.")
    return combined_df

def calculate_and_print_metrics(df: pd.DataFrame):
    """
    Calcula y muestra un resumen de métricas por modelo, distinguiendo entre
    respuestas útiles, respuestas de "no encontrado" y errores de sistema.
    """
    # Frase para identificar respuestas de "no encontrado"
    NOT_FOUND_PHRASE = "No he encontrado esa información"

    grouped = df.groupby('model')
    summary_data = []

    for model_name, group in grouped:
        total_preguntas = len(group)
        
        # Errores de sistema
        errores_sistema = group['sense_score'].isnull()
        num_errores = errores_sistema.sum()
        
        # DataFrame sin los errores de sistema para seguir analizando
        respuestas_validas_df = group[~errores_sistema]
        
        # Respuestas "No Encontrado"
        no_encontrado = respuestas_validas_df['answer'].str.contains(NOT_FOUND_PHRASE, na=False, case=False)
        num_no_encontrado = no_encontrado.sum()
        
        # Respuestas Útiles (el resto)
        respuestas_utiles_df = respuestas_validas_df[~no_encontrado]
        num_utiles = len(respuestas_utiles_df)
        
        # --- Cálculo de Porcentajes ---
        pct_errores = (num_errores / total_preguntas) * 100 if total_preguntas > 0 else 0
        pct_no_encontrado = (num_no_encontrado / total_preguntas) * 100 if total_preguntas > 0 else 0
        pct_utiles = (num_utiles / total_preguntas) * 100 if total_preguntas > 0 else 0
        
        # --- Métricas sobre Respuestas ÚTILES ---
        score_media_utiles = respuestas_utiles_df['sense_score'].mean()
        latencia_media_utiles = respuestas_utiles_df['latency_seconds'].mean()

        summary_data.append({
            'Modelo': model_name,
            'Total': total_preguntas,
            '% Útiles': f"{pct_utiles:.1f}",
            '% No Encontrado': f"{pct_no_encontrado:.1f}",
            '% Errores': f"{pct_errores:.1f}",
            'Score (Útiles)': f"{score_media_utiles:.2f}" if pd.notna(score_media_utiles) else "N/A",
            'Latencia (Útiles)': f"{latencia_media_utiles:.2f}s" if pd.notna(latencia_media_utiles) else "N/A"
        })

    if not summary_data:
        print("🤷 No hay datos para generar un resumen.")
        return

    summary_df = pd.DataFrame(summary_data)
    
    print("\n" + "="*100)
    print("📊 RESUMEN DE MÉTRICAS POR MODELO")
    print("="*100)
    print("  - 'Útiles': Respuestas que aportan información.")
    print("  - 'No Encontrado': Respuestas donde el modelo indica que no halló la información.")
    print("  - 'Errores': Fallos de sistema (timeouts, etc.).")
    print("  - 'Score' y 'Latencia' se calculan solo sobre las respuestas 'Útiles'.")
    print("-" * 100)
    
    print(summary_df.to_string(index=False))
    
    print("="*100)

def main():
    """
    Función principal para ejecutar el proceso de evaluación.
    """
    parser = argparse.ArgumentParser(
        description="Analiza y resume los resultados de las evaluaciones RAG desde archivos CSV."
    )
    parser.add_argument(
        'files',
        nargs='*',
        default=None,
        help="(Opcional) Rutas a los archivos CSV a analizar. Si se omite, se buscarán todos los CSV en 'testing/output/'."
    )
    
    args = parser.parse_args()

    files_to_process = args.files
    
    if not files_to_process:
        search_path = 'testing/output/*.csv'
        print(f"▶️  No se especificaron archivos. Buscando en: '{search_path}'")
        files_to_process = glob.glob(search_path)

    if not files_to_process:
        print("\n🛑 No se encontraron archivos CSV para analizar.")
    else:
        combined_data = load_and_combine_results(files_to_process)
        if combined_data is not None and not combined_data.empty:
            calculate_and_print_metrics(combined_data)

if __name__ == "__main__":
    main()