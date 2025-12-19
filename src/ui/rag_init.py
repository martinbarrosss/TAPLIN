# src/ui/rag_init.py

import streamlit as st
# Importamos las clases necesarias
from src.models.chatbot import RAGChatbot
from src.models.ollama_manager import OllamaModelManager
from src.models.traductor import Traductor             
from src.core.rag_service import RAGMultilingualService 
from src.utils.app_utils import load_vector_store # Necesitas esta importación para cargar la DB

def initialize_rag_system(selected_model: str, temperature: float) -> bool:
    """
    Inicializa la base vectorial, el Traductor y el RAGMultilingualService. 
    Retorna True si la inicialización es exitosa o si ya estaba inicializado.
    """
    # Si el servicio ya existe (guardado como 'chatbot'), simplemente retorna True
    if st.session_state.chatbot is not None: 
        return True 

    # Si no existe, intenta inicializarlo
    with st.spinner(" 🤖 Inicializando sistema RAG multilingüe..."):
        try:
            # 1. Cargar la base de datos vectorial
            vector_store, persist_dir = load_vector_store()
            
            if vector_store is None:
                st.error(
                    f" Base de datos vectorial no encontrada en '{persist_dir}'. "
                    "Por favor, ejecuta el script 'ingest_data.py' primero."
                )
                return False
            
            st.session_state.vector_store = vector_store
            
            # 2. Inicializar el LLM 
            model_manager = OllamaModelManager()
            llm_model = model_manager.create_llm(
                selected_model, 
                temperature=temperature
            )
            
            # 3. Inicializar el Traductor (NUEVO)
            traductor = Traductor() 
            
            # 4. Crear el RAGChatbot base 
            rag_chatbot = RAGChatbot(vector_store, llm_model)
            
            # 5. Crear y guardar el Servicio Multilingüe 
            rag_multilingual_service = RAGMultilingualService(rag_chatbot, traductor)

            st.session_state.chatbot = rag_multilingual_service # Guardamos el servicio como 'chatbot'
            st.success(" 🚀 Sistema RAG Multilingüe listo!")
            return True
        
        except Exception as e:
            st.error(f"Error al inicializar el servicio multilingüe: {str(e)}")
            return False