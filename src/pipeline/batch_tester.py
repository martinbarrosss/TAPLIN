# src/pipeline/batch_tester.py

import os
import pandas as pd
import time
import re
import numpy as np
from dotenv import load_dotenv

# --- Imports del Proyecto ---
from src.models.traductor import Traductor
from src.models.ollama_manager import OllamaModelManager
from src.core.rag_service import RAGMultilingualService
from src.models.chatbot import RAGChatbot
from src.utils.app_utils import load_vector_store

# ==============================================================================
# 1. EL JUEZ AUTOMÁTICO (LLM-as-a-Judge) - VERSIÓN SIMPLIFICADA
# ==============================================================================
class RAGJudge:
    """
    Encapsula la lógica para evaluar la calidad de las respuestas del RAG
    usando un LLM como Juez. Esta versión solo evalúa si la respuesta
    tiene sentido en relación a la pregunta (puntuación binaria 0 o 1).
    """
    def __init__(self, model_name: str, ollama_manager: OllamaModelManager):
        print(f"⚖️  Inicializando Juez con el modelo: {model_name}")
        self.llm = ollama_manager.create_llm(model_name, temperature=0)
    
    def _clean_score(self, score_text: str) -> float:
        """Extrae de forma robusta una puntuación flotante de la respuesta del LLM."""
        match = re.search(r"\b(0|1)\b", score_text) # Solo busca 0 o 1
        if match:
            return float(match.group())
        print(f"⚠️  Advertencia: No se pudo extraer la puntuación (0 o 1) de la salida del Juez: '{score_text}'. Se usará 0.0.")
        return 0.0

    def evaluate(self, question: str, answer: str) -> float:
        """
        Pide al LLM Juez que puntúe si la respuesta tiene sentido para la pregunta.
        Retorna 1 si tiene sentido, 0 si no.
        """
        prompt_sentido = f"""
        Actúa como un juez imparcial. Evalúa si la RESPUESTA tiene sentido en relación a la PREGUNTA.
        La respuesta no tiene que ser perfecta, solo tener una relación lógica.

        PREGUNTA:
        {question}
        RESPUESTA:
        {answer}

        Responde SOLAMENTE con un '1' si la respuesta es relevante y tiene sentido para la pregunta, o con un '0' si no tiene nada que ver o es un error.
        No des explicaciones. Solo el número.
        """
        try:
            score_raw = self.llm.invoke(prompt_sentido).strip()
            return self._clean_score(score_raw)
        except Exception as e:
            print(f"❌ Error en la evaluación del Juez: {e}")
            return 0.0

# ==============================================================================
# 2. FUNCIÓN ORQUESTADORA DE LA EVALUACIÓN
# ==============================================================================
def run_evaluation_batch(input_filename: str, output_filename: str, models_to_test: list[str], judge_model: str):
    """
    Orquesta todo el proceso de evaluación. Es llamada por el script de entrada (sim_test.py).
    """
    load_dotenv()
    INPUT_DIR = "testing/input"
    OUTPUT_DIR = "testing/output"
    input_path = os.path.join(INPUT_DIR, input_filename)
    output_path = os.path.join(OUTPUT_DIR, output_filename)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print("🚀 Iniciando el Orquestador de Evaluaciones RAG...")
    
    try:
        print("🛠️  Cargando componentes base (VectorDB, Traductor, Ollama Manager)...")
        vector_store, _ = load_vector_store()
        if vector_store is None: raise FileNotFoundError("No se pudo cargar la base de datos Chroma.")
        traductor = Traductor()
        ollama_manager = OllamaModelManager()
        juez = RAGJudge(judge_model, ollama_manager)
        
        print(f"📄 Cargando preguntas desde '{input_path}'...")
        if not os.path.exists(input_path): raise FileNotFoundError(f"El archivo de entrada '{input_path}' no existe.")
        df_input = pd.read_csv(input_path)
        if 'query' not in df_input.columns: raise ValueError("El archivo de entrada debe tener una columna 'query'.")
        if 'language' not in df_input.columns: df_input['language'] = 'es'
        print(f"✅ {len(df_input)} preguntas cargadas.")
        
    except (FileNotFoundError, ValueError) as e:
        print(f"🛑 Error crítico en la inicialización: {e}")
        return

    results_data = []
    total_iterations = len(models_to_test) * len(df_input)
    current_iteration = 0
    
    for model_name in models_to_test:
        print(f"\n--- 🤖 PROBANDO MODELO: {model_name} ---")
        try:
            llm = ollama_manager.create_llm(model_name, temperature=0.1)
            rag_chatbot = RAGChatbot(vector_store=vector_store, llm=llm)
            rag_service = RAGMultilingualService(
                chatbot=rag_chatbot,
                traductor=traductor
            )
            for index, row in df_input.iterrows():
                current_iteration += 1
                question, lang = row['query'], row['language']
                print(f"  [{current_iteration}/{total_iterations}] Pregunta ({lang}): {question[:60]}...")
                start_time = time.time()
                try:
                    response_dict = rag_service.answer_question(question, lang)
                    answer = response_dict.get('answer', 'ERROR: No se generó respuesta.')
                    sources = response_dict.get('sources', [])
                    context_text = "\n\n".join([doc.page_content for doc in sources])
                    source_names = [doc.metadata.get("source", "Unknown") for doc in sources]
                    
                    # --- EVALUACIÓN SIMPLIFICADA ---
                    sense_score = juez.evaluate(question, answer)
                    latency = round(time.time() - start_time, 2)
                    
                    results_data.append({
                        "model": model_name, "question": question, "language": lang,
                        "answer": answer, "sense_score": sense_score,
                        "latency_seconds": latency,
                        "context": context_text, "sources": str(source_names)
                    })
                except Exception as e:
                    print(f"    ❌ Error procesando pregunta: {e}")
                    results_data.append({
                        "model": model_name, "question": question, "language": lang,
                        "answer": f"ERROR: {e}", "sense_score": np.nan,
                        "latency_seconds": np.nan, "context": "", "sources": ""
                    })
        except Exception as e:
            print(f"  ❌❌ Error crítico cargando el modelo {model_name}: {e}.")
            for _, row in df_input.iterrows():
                 results_data.append({
                    "model": model_name, "question": row['query'], "language": row.get('language', 'es'),
                    "answer": f"FATAL_MODEL_ERROR: {e}", "sense_score": np.nan,
                    "latency_seconds": np.nan, "context": "", "sources": ""
                })

    if not results_data:
        print("\n⚠️ No se generó ningún resultado.")
        return
        
    print(f"\n✅ Proceso finalizado. Guardando {len(results_data)} resultados en '{output_path}'...")
    df_results = pd.DataFrame(results_data)
    df_results.to_csv(output_path, index=False, sep=';', encoding='utf-8-sig')
    
    print("\n📊 Resumen de Puntuaciones Medias por Modelo:")
    # Resumen actualizado para la nueva puntuación
    if "sense_score" in df_results.columns:
        summary = df_results.groupby("model")[["sense_score", "latency_seconds"]].mean().round(3)
        print(summary)
    else:
        print("No se encontró la columna 'sense_score' para generar el resumen.")

    print("\n¡Evaluación completada con éxito!")

if __name__ == "__main__":
    import argparse
    
    # --- LISTA DE MODELOS DISPONIBLES ---
    AVAILABLE_MODELS = [
        "gpt-oss:20b-cloud",
        "deepseek-v3.1:671b-cloud",
        "kimi-k2:1t-cloud",
        "qwen3-coder:480b-cloud"
    ]

    parser = argparse.ArgumentParser(
        description="Ejecuta un batch de evaluación de modelos RAG con un Juez LLM.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--input", default="PromptsFrigo.csv", help="Archivo de entrada CSV en 'testing/input'.")
    parser.add_argument("--output", default=f"evaluation_results_{int(time.time())}.csv", help="Archivo de salida CSV en 'testing/output'.")
    parser.add_argument("--model", default=None, help="(Opcional) Especifica un modelo a probar, saltando el menú interactivo.")
    parser.add_argument("--judge-model", default="gpt-oss:20b-cloud", help="Modelo LLM que actuará como Juez.")
    
    args = parser.parse_args()
    
    model_to_evaluate = None

    if args.model:
        if args.model in AVAILABLE_MODELS:
            model_to_evaluate = args.model
        else:
            print(f"🛑 Error: El modelo '{args.model}' no está en la lista de modelos disponibles.")
            print("   Modelos disponibles:", ", ".join(AVAILABLE_MODELS))
            exit(1)
    else:
        # --- MENÚ INTERACTIVO ---
        print("\n--- Por favor, elige un modelo para evaluar ---")
        for i, model_name in enumerate(AVAILABLE_MODELS):
            print(f"  [{i+1}] {model_name}")
        
        while True:
            try:
                choice = int(input(f"Introduce el número del modelo (1-{len(AVAILABLE_MODELS)}): ")) - 1
                if 0 <= choice < len(AVAILABLE_MODELS):
                    model_to_evaluate = AVAILABLE_MODELS[choice]
                    break
                else:
                    print("Opción no válida. Por favor, introduce un número de la lista.")
            except ValueError:
                print("Entrada no válida. Por favor, introduce un número.")

    run_evaluation_batch(
        input_filename=args.input, 
        output_filename=args.output,
        models_to_test=[model_to_evaluate], # La función espera una lista
        judge_model=args.judge_model
    )