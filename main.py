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


# --- Funciones Auxiliares (DEFINIDAS ANTES DE SER LLAMADAS) ---

def configure_page_and_header():
    """Configura el título de la página y el encabezado principal de la aplicación."""
    st.set_page_config(
        page_title="Chatbot RAG - Manuales",
        page_icon="🤖",
        layout="wide"
    )
    st.title(" Asistente de Manuales de Electrodomésticos")
    st.markdown(
        "Haz preguntas específicas sobre los manuales y recibe respuestas "
        "precisas basadas en el contenido real."
    )

def setup_app_state():
    """Carga variables de entorno, inicializa el estado de sesión y procesa la sidebar."""
    load_dotenv()
    initialize_session_state()
    
    # Configura la barra lateral (sidebar) y guarda la temperatura y el idioma
    selected_model, new_temp, new_lang = setup_sidebar(
        st.session_state.selected_model, 
        st.session_state.temperature,
        st.session_state.language 
    )
    st.session_state.selected_model = selected_model
    st.session_state.temperature = new_temp 
    st.session_state.language = new_lang # GUARDAMOS el idioma actualizado

# --- Función Principal de Flujo (Orquestación) ---

def run_app():
    """
    Función principal que define el flujo secuencial de la aplicación Streamlit.
    """
    # 1. Configuración y Estado (Ahora las funciones están definidas)
    configure_page_and_header() 
    setup_app_state() 

    # 2. Inicialización del RAG (Llamada a la función externa)
    # Inicializa el RAGMultilingualService
    rag_ready = initialize_rag_system(
        st.session_state.selected_model, 
        st.session_state.temperature
    )

    if not rag_ready: 
        return

    # 3. Definición de la estructura de pestañas
    tab_chat, tab_info, tab_historial = st.tabs(["💬 Chat", "ℹ️ Información", "📜 Historial"])

    # 4. Renderizado de pestañas
    with tab_chat:
        render_tab_chat() 

    with tab_info:
        render_tab_info()

    with tab_historial:
        render_tab_history()


if __name__ == "__main__":
    run_app()