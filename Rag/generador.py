import os
import torch # Importar torch para DEVICE
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline

# Definimos el dispositivo de hardware para usar en LLM
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def inicializador_rag(CHROMA_PERSIST_DIR, EMBEDDING_MODEL_NAME):

    print(f"--- Cargando modelo de Embedding ({EMBEDDING_MODEL_NAME}) ---")
    embeddings = HuggingFaceBgeEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )

    print(f"--- Cargando Base de Datos Vectorial (ChromaDB) ---")
    # Cargamos el vector persistente
    vector_store = Chroma(
        persist_directory=CHROMA_PERSIST_DIR, 
        embedding_function=embeddings
    )
    
    return vector_store, embeddings

# El bucle de preguntas se manejará en main.py.
def inicializar_qa_chain(vector_store: Chroma, LLM_MODEL_NAME: str):

    print(f"--- Cargando el LLM Generativo ({LLM_MODEL_NAME}) en {DEVICE} ---")

    tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        LLM_MODEL_NAME,
        device_map="auto"
    )

    # Creamos un pipeline de texto con el modelo y el tokenizador
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=512, # Más tokens para chat interactivo
        temperature=0.1, 
        trust_remote_code=True
    )

    llm = HuggingFacePipeline(pipeline=pipe)

    # PROMPT CRUCIAL CHATTT (habrá que mejorarlo, se puede usar gemini que hace buenos prompts)
    prompt_template = """
    Eres un asistente experto, llamado NeveraBot, especializado en el modelo de frigorífico/congelador (Nevera).
    Tu tarea es responder a las preguntas del usuario **ÚNICAMENTE** basándote en el siguiente contexto extraído del manual.
    Si el contexto no proporciona la información, debes indicar amablemente que la información no está en el manual. Sé conciso y claro.

    Contexto: {context}

    Pregunta del Usuario: {question}

    Respuesta:
    """
    RAG_PROMPT = PromptTemplate(
        template=prompt_template, 
        input_variables=["context", "question"]
    )

    # Unimos el retriever (Base Vectorial) con el LLM y el Prompt.
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff", 
        retriever=vector_store.as_retriever(search_kwargs={"k": 4}), # Aumentamos a 4 fragmentos
        chain_type_kwargs={"prompt": RAG_PROMPT},
        return_source_documents=True 
    )
    
    print("-" * 70)
    print("✅ Módulos LLM y QA Chain inicializados.")
    print("-" * 70)
    
    # Devolvemos la cadena RAG lista para ser usada en el bucle.
    return qa_chain