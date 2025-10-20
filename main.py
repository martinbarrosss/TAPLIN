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
    
    selected_model, temperature = setup_sidebar(
        st.session_state.selected_model, 0.7
    )
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
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
                        llm = model_manager.create_llm(selected_model, temperature=temperature)
                        st.session_state.chatbot = RAGChatbot(vector_store, llm)
                        st.success(" Sistema RAG inicializado correctamente")
                
                except Exception as e:
                    st.error(f"Error al inicializar: {str(e)}")
        
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
    
    with col2:
        st.subheader(" Información")
        
        st.metric("Modelo activo", st.session_state.selected_model)
        st.metric("Temperatura", f"{temperature:.1f}")
        
        if st.session_state.chat_history:
            st.metric("Mensajes en sesión", len(st.session_state.chat_history))
        
        if st.button(" Limpiar historial"):
            st.session_state.chat_history = []
            st.rerun()
        
        st.markdown("---")
        st.markdown(
            "**Modelo RAG:**\n"
            "- ChromaDB para almacenamiento vectorial\n"
            "- LangChain para orquestación\n"
            "- Ollama para inferencia (Local/Cloud)"
        )

if __name__ == "__main__":
    main()