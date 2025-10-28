import streamlit as st
# Necesitas estas importaciones para crear el chatbot y el LLM
from src.models.chatbot import RAGChatbot
from src.models.ollama_manager import OllamaModelManager
from src.utils.app_utils import load_vector_store # Necesitas esta importación para cargar la DB

def initialize_rag_system(selected_model: str, temperature: float) -> bool:
    """
    Inicializa la base vectorial y el objeto RAGChatbot. 
    Se ejecuta solo si el chatbot aún no está en el estado de sesión.
    Retorna True si la inicialización es exitosa o si ya estaba inicializado.
    """
    # Si el chatbot ya existe, simplemente retorna True
    if st.session_state.chatbot is not None:
        return True 

    # Si no existe, intenta inicializarlo
    with st.spinner(" Inicializando sistema RAG..."):
        try:
            vector_store, persist_dir = load_vector_store()
            
            if vector_store is None:
                st.error(
                    f" Base de datos vectorial no encontrada en '{persist_dir}'. "
                    "Por favor, ejecuta el script 'ingest_data.py' primero."
                )
                return False # Falla la carga
            
            st.session_state.vector_store = vector_store
            
            model_manager = OllamaModelManager()
            llm = model_manager.create_llm(
                selected_model, 
                temperature=temperature
            )
            
            # Crea y guarda el objeto RAGChatbot en el estado de sesión
            st.session_state.chatbot = RAGChatbot(vector_store, llm)
            st.success(" Sistema RAG inicializado correctamente")
            return True # Inicialización exitosa
        
        except Exception as e:
            st.error(f"Error al inicializar: {str(e)}")
            return False
