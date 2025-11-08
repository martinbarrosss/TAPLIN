# src/ui/app_ui.py

import streamlit as st
import os

from src.models.ollama_manager import OllamaModelManager

def initialize_session_state():
    """Inicializa variables de sesión de Streamlit."""
    if "vector_store" not in st.session_state:
        st.session_state.vector_store = None
    if "chatbot" not in st.session_state:
        st.session_state.chatbot = None
    if "selected_model" not in st.session_state:
        st.session_state.selected_model = "mistral"
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "temperature" not in st.session_state:
        st.session_state.temperature = 0.1
    # --- NUEVA VARIABLE DE ESTADO ---
    if "language" not in st.session_state:
        st.session_state.language = "es" # Idioma por defecto: Español


def setup_sidebar(selected_model: str, temperature: float, current_language: str):
    """Configura la barra lateral de Streamlit (Modelo, Temperatura e Idioma)."""
    
    with st.sidebar:
        st.title(" Configuración")
        
        # 1. Selector de Modelo 
        model_manager = OllamaModelManager()
        available_models = model_manager.get_available_models()
        
        selected_model = st.selectbox(
            "Selecciona el modelo Ollama:",
            options=available_models,
            index=available_models.index(selected_model)
            if selected_model in available_models else 0
        )
        st.session_state.selected_model = selected_model
        
        st.markdown("---")
        
        # 2. Selector de Idioma 
        selected_option = st.selectbox(
            "Idioma de la conversación:",
            options=["Español (es)", "Gallego (gl)"],
            index=0 if current_language == "es" else 1,
            format_func=lambda x: x.split(" ")[0] # Muestra solo el nombre
        )
        # Extrae el código ('es' o 'gl') de la opción seleccionada
        new_language = selected_option.split("(")[1].replace(")", "").strip() 
        
        st.session_state.language = new_language
        
        st.markdown("---")

        # 3. Slider de Temperatura
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
        
        # Retorna los 3 valores actualizados: modelo, temperatura y nuevo idioma
        return selected_model, temperature, new_language