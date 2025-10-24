import streamlit as st
import os

from src.models.ollama_manager import OllamaModelManager

def initialize_session_state():

    pass


def setup_sidebar(selected_model: str, temperature: float):
    
    """Configura la barra lateral de Streamlit (Modelo y Temperatura)."""
    with st.sidebar:
        st.title(" Configuración")
        
        model_manager = OllamaModelManager() #instancia del gestor de modelos
        available_models = model_manager.get_available_models()
        
        selected_model = st.selectbox(
            "Selecciona el modelo Ollama:",
            options=available_models,
            index=available_models.index(selected_model)
            if selected_model in available_models else 0
        )
        
        st.session_state.selected_model = selected_model
        
        st.markdown("---")
        
        temperature = st.slider(
            "Temperatura (creatividad del modelo):",
            min_value=0.0,
            max_value=1.0,
            value=temperature,
            step=0.1
        )
        
        st.markdown("---")
        
        st.info(
            " **Información**\n\n"
            "Este chatbot utiliza RAG para responder preguntas "
            "basadas en manuales de electrodomésticos.\n\n"
        )
        
        return selected_model, temperature