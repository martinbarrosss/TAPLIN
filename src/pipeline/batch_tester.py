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
# 1. EL JUEZ AUTOMÁTICO (LLM-as-a-Judge)
# ==============================================================================
class RAGJudge:
    def __init__(self, model_name: str, ollama_manager: OllamaModelManager):
        print(f"Inicializando Juez con el modelo: {model_name}")
        self.llm = ollama_manager.create_llm(model_name, temperature=0)
    
    def _clean_score(self, score_text: str) -> float:
        match = re.search(r"\b(0|1)\b", score_text)
        if match:
            return float(match.group())
        print(f"Advertencia: No se pudo extraer la puntuación (0 o 1). Se usará 0.0.")
        return 0.0

    def evaluate(self, question: str, answer: str) -> float:
        prompt_sentido = f"""
        Actúa como un juez imparcial. Evalúa si la RESPUESTA tiene sentido en relación a la PREGUNTA.
        La respuesta no tiene que ser perfecta, solo tener una relación lógica.

        PREGUNTA: {question}
        RESPUESTA: {answer}

        Responde SOLAMENTE con un '1' si la respuesta es relevante, o con un '0' si no tiene sentido.
        """
        try:
            score_raw = self.llm.invoke(prompt_sentido).strip()
            return self._clean_score(score_raw)
        except Exception as e:
            print(f"Error en la evaluación del Juez: {e}")
            return 0.0

# ==============================================================================
# 2. FUNCIÓN ORQUESTADORA (CON REINTENTOS ANTI-503)
# ==============================================================================
def run_evaluation_batch(input_filename: str, output_filename: str, models_to_test: list[str], judge_model: str):
    load_dotenv()
    INPUT_DIR = "testing/input"
    OUTPUT_DIR = "testing/output"
    input_path = os.path.join(INPUT_DIR, input_filename)
    output_path = os.path.join(OUTPUT_DIR, output_filename)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print("Iniciando el Orquestador de Evaluaciones RAG...")
    
    try:
        print("Cargando componentes base...")
        vector_store, _ = load_vector_store()
        if vector_store is None: raise FileNotFoundError("No se pudo cargar la base de datos Chroma.")
        traductor = Traductor()
        ollama_manager = OllamaModelManager()
        juez = RAGJudge(judge_model, ollama_manager)
        
        if not os.path.exists(input_path): raise FileNotFoundError(f"Archivo '{input_path}' no existe.")
        df_input = pd.read_csv(input_path)
        if 'query' not in df_input.columns: raise ValueError("Falta columna 'query' en CSV.")
        if 'language' not in df_input.columns: df_input['language'] = 'es'
        print(f"{len(df_input)} preguntas cargadas.")
        
    except Exception as e:
        print(f"Error crítico inicialización: {e}")
        return

    results_data = []
    total_iterations = len(models_to_test) * len(df_input)
    current_iteration = 0
    
    for model_name in models_to_test:
        print(f"\n--- PROBANDO MODELO: {model_name} ---")
        try:
            llm = ollama_manager.create_llm(model_name, temperature=0.1)
            rag_chatbot = RAGChatbot(vector_store=vector_store, llm=llm)
            rag_service = RAGMultilingualService(chatbot=rag_chatbot, traductor=traductor)
            
            for index, row in df_input.iterrows():
                current_iteration += 1
                question, lang = row['query'], row['language']
                
                # --- SISTEMA DE REINTENTOS ---
                max_retries = 3
                success = False
                
                for attempt in range(max_retries):
                    try:
                        attempt_label = f"(Intento {attempt+1})" if attempt > 0 else ""
                        print(f"  [{current_iteration}/{total_iterations}] {attempt_label} Pregunta ({lang}): {question[:50]}...")
                        start_time = time.time()
                        
                        # 1. RAG
                        response_dict = rag_service.answer_question(question, lang)
                        answer = response_dict.get('answer', 'ERROR: Sin respuesta')
                        sources = response_dict.get('sources', [])
                        
                        # 2. Juez
                        sense_score = juez.evaluate(question, answer)
                        latency = round(time.time() - start_time, 2)
                        
                        # Guardar datos
                        context_text = "\n\n".join([doc.page_content for doc in sources])
                        source_names = [doc.metadata.get("source", "Unknown") for doc in sources]
                        
                        results_data.append({
                            "model": model_name, "question": question, "language": lang,
                            "answer": answer, "sense_score": sense_score,
                            "latency_seconds": latency,
                            "context": context_text, "sources": str(source_names)
                        })
                        success = True
                        break # ¡Éxito! Salimos del bucle de intentos

                    except Exception as e:
                        error_msg = str(e)
                        # Si es error 503 o conexión, esperamos
                        if ("503" in error_msg or "Connection refused" in error_msg) and attempt < max_retries - 1:
                            wait_s = (attempt + 1) * 15 # Espera 15s, 30s...
                            print(f"    Servidor saturado ({error_msg}). Esperando {wait_s}s...")
                            time.sleep(wait_s)
                        else:
                            print(f"  Error en intento {attempt+1}: {e}")
                            if attempt == max_retries - 1: # Último intento fallido
                                results_data.append({
                                    "model": model_name, "question": question, "language": lang,
                                    "answer": f"ERROR FINAL: {e}", "sense_score": 0.0,
                                    "latency_seconds": 0.0, "context": "", "sources": ""
                                })

        except Exception as e:
            print(f"  Error crítico cargando modelo {model_name}: {e}")

    if results_data:
        print(f"\n Guardando {len(results_data)} resultados en '{output_path}'...")
        df_results = pd.DataFrame(results_data)
        df_results.to_csv(output_path, index=False, sep=';', encoding='utf-8-sig')
        if "sense_score" in df_results.columns:
            print(df_results.groupby("model")[["sense_score", "latency_seconds"]].mean().round(3))

if __name__ == "__main__":
    import argparse
    AVAILABLE_MODELS = ["gpt-oss:20b-cloud", "deepseek-v3.1:671b-cloud", "kimi-k2:1t-cloud", "qwen3-coder:480b-cloud"]

    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--input", default="PromptsFrigo.csv", help="Input CSV")
    parser.add_argument("--output", default=f"results_{int(time.time())}.csv", help="Output CSV")
    parser.add_argument("--model", default=None, help="Modelo específico")
    parser.add_argument("--judge-model", default="gpt-oss:20b-cloud", help="Modelo Juez")
    
    args = parser.parse_args()
    
    model_to_evaluate = None
    if args.model:
        if args.model in AVAILABLE_MODELS: model_to_evaluate = args.model
        else: exit(f"Error: Modelo {args.model} no disponible.")
    else:
        print("\nModelos disponibles:")
        for i, m in enumerate(AVAILABLE_MODELS): print(f"  [{i+1}] {m}")
        try:
            choice = int(input(f"Elige modelo (1-{len(AVAILABLE_MODELS)}): ")) - 1
            model_to_evaluate = AVAILABLE_MODELS[choice]
        except: exit("Opción inválida.")

    run_evaluation_batch(args.input, args.output, [model_to_evaluate], args.judge_model)