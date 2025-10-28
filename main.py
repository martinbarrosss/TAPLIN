import streamlit as st
from dotenv import load_dotenv

# --- Importaciones de tu proyecto (lógica de negocio) ---
from src.models.chatbot import RAGChatbot
from src.models.ollama_manager import OllamaModelManager
# --- Importación de utilidades (funciones auxiliares) ---
from src.utils.app_utils import load_vector_store
from src.ui.app_ui import setup_sidebar, initialize_session_state

def main():
    """Función principal de la aplicación Streamlit: Inicialización y Loop de Chat."""
    load_dotenv()
    initialize_session_state()
    
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
    
    # Configura la barra lateral y guarda la temperatura en el estado de sesión
    selected_model, new_temp = setup_sidebar(
        st.session_state.selected_model, 
        st.session_state.temperature  # Lee la temperatura del estado
    )
    
    # Guarda la temperatura seleccionada en el slider de vuelta al estado
    st.session_state.temperature = new_temp 
    
    # Define las pestañas en lugar de las columnas
    tab_chat, tab_info, tab_historial = st.tabs(["💬 Chat", "ℹ️ Información", "📜 Historial"])
    
    # Pestaña de Chat
    with tab_chat:

        st.subheader("Chat")
        
        if st.session_state.chatbot is None:
            with st.spinner(" Inicializando sistema RAG..."):
                try:
                    vector_store, persist_dir = load_vector_store()
                    
                    if vector_store is None:
                        st.error(
                            f" Base de datos vectorial no encontrada en '{persist_dir}'. "
                            "Por favor, ejecuta el script 'ingest_data.py' primero."
                        )
                    else:
                        st.session_state.vector_store = vector_store
                        
                        model_manager = OllamaModelManager()
                        # Usa la temperatura del estado de sesión
                        llm = model_manager.create_llm(
                            selected_model, 
                            temperature=st.session_state.temperature
                        )
                        st.session_state.chatbot = RAGChatbot(vector_store, llm)
                        st.success(" Sistema RAG inicializado correctamente")
                
                except Exception as e:
                    st.error(f"Error al inicializar: {str(e)}")
        
        # Historial de chat (visualización principal)
        for msg in st.session_state.chat_history:

            st.chat_message(msg["role"]).write(msg["content"])
        
        user_question = st.chat_input("Escribe tu pregunta sobre los manuales...", key="user_input")
        
        if user_question and st.session_state.chatbot:
            st.session_state.chat_history.append({"role": "user", "content": user_question})
            st.chat_message("user").write(user_question)
            
            with st.spinner(" Buscando información..."):
                try:
                    response = st.session_state.chatbot.answer_question(user_question)
                    
                    st.chat_message("assistant").write(response["answer"])
                    
                    if response["sources"]:
                        with st.expander(" Fuentes consultadas"):
                            for source in set(response["sources"]):
                                st.caption(f" {source}")
                    
                    st.session_state.chat_history.append({"role": "assistant", "content": response["answer"]})
                    
                except Exception as e:
                    st.error(f"Error al procesar: {str(e)}")
    
    # Pestaña de Información
    with tab_info:
        st.subheader("Información del Sistema")
        
        st.metric("Modelo activo", st.session_state.selected_model)
        st.metric("Temperatura", f"{st.session_state.temperature:.1f}")
        
        if st.session_state.chat_history:
            st.metric("Mensajes en sesión", len(st.session_state.chat_history))

        st.markdown("---")
        st.markdown(
            "**Modelo RAG:**\n"
            "- ChromaDB para almacenamiento vectorial\n"
            "- LangChain para orquestación\n"
            "- Ollama para inferencia (Local/Cloud)"
        )

    # Pestaña de Historial
    with tab_historial:

        st.subheader("Revisar Historial de la Sesión")

        if st.button("🗑️ Limpiar historial"):
            st.session_state.chat_history = []
            st.rerun()
        
        st.markdown("---")

        if not st.session_state.chat_history:
            st.caption("El historial de esta sesión está vacío.")
        else:
            # Iteramos y mostramos de forma compacta (como en tu expander original)
            for msg in st.session_state.chat_history:
                if msg["role"] == "user":
                    st.markdown(f"**Tú:** {msg['content']}")
                else:
                    st.markdown(f"**Asistente:** {msg['content']}")
                st.markdown("---") # Separador

if __name__ == "__main__":
    main()