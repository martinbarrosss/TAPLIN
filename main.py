# main.py
# MODIFICADO

import streamlit as st
from dotenv import load_dotenv

from src.ui.app_ui import setup_sidebar, initialize_session_state 
from src.ui.rag_init import initialize_rag_system 
from src.ui.renders import render_tab_chat, render_tab_info, render_tab_history
from src.ui.i18n import get_translator 


def configure_page_and_header(t): 
    """Configura el título de la página y el encabezado principal de la aplicación."""
    st.set_page_config(
        page_title=t("page_title"), 
        page_icon="🤖",
        layout="wide"
    )
    st.title(t("header_title")) 
    st.markdown(t("header_markdown")) 

def setup_app_state_and_sidebar():
    """
    Carga variables de entorno, inicializa el estado de sesión, 
    procesa la sidebar y retorna el traductor 't' correcto.
    """
    load_dotenv()
    initialize_session_state()
    
    t = get_translator(st.session_state.language)
    
    # Configura la barra lateral (sidebar)
    selected_model, new_temp, new_lang = setup_sidebar(
        st.session_state.selected_model, 
        st.session_state.temperature,
        st.session_state.language,
        t # <-- Pasamos 't' para que la sidebar se traduzca
    )
    
    # Actualizar el estado de sesión
    st.session_state.selected_model = selected_model
    st.session_state.temperature = new_temp 
    
    # Comprobamos si el idioma cambió
    if st.session_state.language != new_lang:
        st.session_state.language = new_lang
        st.rerun() 
    
    # Retornamos el traductor 't' (que está actualizado o estaba bien)
    return t

# --- Función Principal de Flujo  ---
def run_app():
    """
    Función principal que define el flujo secuencial de la aplicación Streamlit.
    """
    t = setup_app_state_and_sidebar() 
    
    configure_page_and_header(t) 

    # Inicialización del RAG 
    rag_ready = initialize_rag_system(
        st.session_state.selected_model, 
        st.session_state.temperature
    )

    if not rag_ready: 
        return

    # Definición de la estructura de pestañas
    tab_chat, tab_info, tab_historial = st.tabs([
        t("tab_chat"), 
        t("tab_info"), 
        t("tab_history")
    ])

    # Renderizado de pestañas
    with tab_chat:
        render_tab_chat(t)

    with tab_info:
        render_tab_info(t) 

    with tab_historial:
        render_tab_history(t) 


if __name__ == "__main__":
    run_app()