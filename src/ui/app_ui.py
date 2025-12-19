# src/ui/app_ui.py
# MODIFICADO

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
    if "language" not in st.session_state:
        st.session_state.language = "es" # Idioma por defecto: Español
    # Variable para gestionar las fuentes del último mensaje
    if "last_sources" not in st.session_state:
        st.session_state.last_sources = []


def setup_sidebar(selected_model: str, temperature: float, current_language: str, t):
    """Configura la barra lateral de Streamlit (Modelo, Temperatura e Idioma)."""
    
    with st.sidebar:
        st.title(t("sidebar_title"))
        
        # 1. Selector de Modelo 
        model_manager = OllamaModelManager()
        available_models = model_manager.get_available_models()
        
        selected_model = st.selectbox(
            t("sidebar_model_select"),
            options=available_models,
            index=available_models.index(selected_model)
            if selected_model in available_models else 0
        )
        
        st.markdown("---")
        
        # 2. Selector de Idioma
        # Usamos las claves 'es' y 'gl' para la lógica, y 't' para el display
        lang_options_display = {
            "es": t("sidebar_lang_es"),
            "gl": t("sidebar_lang_gl")
        }
        lang_keys = list(lang_options_display.keys())
        lang_display_names = list(lang_options_display.values())

        selected_display_name = st.selectbox(
            t("sidebar_lang_select"),
            options=lang_display_names,
            index=lang_keys.index(current_language), # Usa el 'es' o 'gl' actual
            format_func=lambda x: x.split(" (")[0] # Muestra solo "Español" o "Gallego"
        )
        
        # Encontramos la clave ('es' o 'gl') basada en el nombre mostrado
        new_language = lang_keys[lang_display_names.index(selected_display_name)]
        
        st.markdown("---")

        # 3. Slider de Temperatura
        temperature = st.slider(
            t("sidebar_temp_slider"),
            min_value=0.0,
            max_value=1.0,
            value=temperature, 
            step=0.1
        )
        
        st.markdown("---")
        
        st.info(
            f" **{t('sidebar_info_title')}**\n\n"
            f"{t('sidebar_info_body')}\n\n"
        )
        
        # Retorna los 3 valores actualizados
        return selected_model, temperature, new_language