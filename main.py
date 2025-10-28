import streamlit as st
from dotenv import load_dotenv

# --- Importaciones de tu proyecto (lógica de negocio) ---
from src.models.chatbot import RAGChatbot
from src.models.ollama_manager import OllamaModelManager
# --- Importación de utilidades (funciones auxiliares) ---
from src.ui.app_ui import setup_sidebar, initialize_session_state 
from src.ui.rag_init import initialize_rag_system 
from src.ui.renders import render_tab_chat, render_tab_info, render_tab_history

# --- Funciones Auxiliares (Separadas para la legibilidad del flujo principal) ---

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
    
    # Configura la barra lateral (sidebar) y guarda la temperatura
    selected_model, new_temp = setup_sidebar(
        st.session_state.selected_model, 
        st.session_state.temperature
    )
    st.session_state.selected_model = selected_model
    st.session_state.temperature = new_temp 


# Función Principal de Flujo (Orquestación) ---

def run_app():
    """
    Función principal que define el flujo secuencial de la aplicación Streamlit.
    """
    # 1. Configuración y Estado
    setup_app_state()          
    configure_page_and_header() 
    
    # 2. Inicialización del RAG (Llamada a la función externa)
    rag_ready = initialize_rag_system(
        st.session_state.selected_model, 
        st.session_state.temperature
    )

    if not rag_ready and st.session_state.chatbot is None:
        return

    # 3. Definición de la estructura de pestañas
    tab_chat, tab_info, tab_historial = st.tabs(["💬 Chat", "ℹ️ Información", "📜 Historial"])

    # 4. Renderizado de pestañas -- codigo en ui/renders.py
    with tab_chat:
        render_tab_chat()

    with tab_info:
        render_tab_info()

    with tab_historial:
        render_tab_history()


if __name__ == "__main__":
    run_app()
