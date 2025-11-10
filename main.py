# main.py
# MODIFICADO

import streamlit as st
from dotenv import load_dotenv

# --- Importaciones de tu proyecto (lógica de negocio) ---
from src.models.ollama_manager import OllamaModelManager
from src.models.chatbot import RAGChatbot # Mantenemos para tipado
from src.models.traductor import Traductor # NUEVA IMPORTACIÓN
from src.core.rag_service import RAGMultilingualService # NUEVA IMPORTACIÓN

# --- Importación de utilidades (funciones auxiliares) ---
from src.ui.app_ui import setup_sidebar, initialize_session_state 
from src.ui.rag_init import initialize_rag_system 
from src.ui.renders import render_tab_chat, render_tab_info, render_tab_history
# --- NUEVA IMPORTACIÓN ---
from src.ui.i18n import get_translator 


# --- Funciones Auxiliares (MODIFICADAS) ---

def configure_page_and_header(t): # Acepta 't'
    """Configura el título de la página y el encabezado principal de la aplicación."""
    st.set_page_config(
        page_title=t("page_title"), # <-- MODIFICADO
        page_icon="🤖",
        layout="wide"
    )
    st.title(t("header_title")) # <-- MODIFICADO
    st.markdown(t("header_markdown")) # <-- MODIFICADO

def setup_app_state_and_sidebar():
    """
    Carga variables de entorno, inicializa el estado de sesión, 
    procesa la sidebar y retorna el traductor 't' correcto.
    """
    load_dotenv()
    initialize_session_state()
    
    # 1. Obtenemos el traductor 't' basado en el idioma *actual* # (para traducir la propia sidebar)
    t = get_translator(st.session_state.language)
    
    # 2. Configura la barra lateral (sidebar)
    selected_model, new_temp, new_lang = setup_sidebar(
        st.session_state.selected_model, 
        st.session_state.temperature,
        st.session_state.language,
        t # <-- Pasamos 't' para que la sidebar se traduzca
    )
    
    # 3. Actualizamos el estado de sesión
    st.session_state.selected_model = selected_model
    st.session_state.temperature = new_temp 
    
    # 4. Comprobamos si el idioma cambió
    if st.session_state.language != new_lang:
        st.session_state.language = new_lang
        # Si cambió, recargamos la app para que TODO se traduzca
        st.rerun() 
    
    # 5. Retornamos el traductor 't' (que está actualizado o estaba bien)
    return t

# --- Función Principal de Flujo (Orquestación) ---

def run_app():
    """
    Función principal que define el flujo secuencial de la aplicación Streamlit.
    """
    # 1. Configuración y Estado (MODIFICADO)
    # Esta función ahora maneja la sidebar y devuelve el 't' correcto
    t = setup_app_state_and_sidebar() 
    
    # 2. Configuración de la Página (ahora usa el 't' correcto)
    configure_page_and_header(t) 

    # 3. Inicialización del RAG (Llamada a la función externa)
    # Inicializa el RAGMultilingualService
    rag_ready = initialize_rag_system(
        st.session_state.selected_model, 
        st.session_state.temperature
    )

    if not rag_ready: 
        return

    # 4. Definición de la estructura de pestañas (MODIFICADO)
    tab_chat, tab_info, tab_historial = st.tabs([
        t("tab_chat"), 
        t("tab_info"), 
        t("tab_history")
    ])

    # 5. Renderizado de pestañas (MODIFICADO)
    with tab_chat:
        render_tab_chat(t) # <-- Pasa 't'

    with tab_info:
        render_tab_info(t) # <-- Pasa 't'

    with tab_historial:
        render_tab_history(t) # <-- Pasa 't'


if __name__ == "__main__":
    run_app()