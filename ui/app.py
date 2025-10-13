import streamlit as st
from streamlit_option_menu import option_menu

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


def setup_sidebar():
    """Configura la barra lateral de Streamlit."""
    with st.sidebar:
        st.title("⚙️ Configuración")
        
        # Selector de modelo
        model_manager = OllamaModelManager()
        available_models = model_manager.get_available_models()
        
        selected_model = st.selectbox(
            "Selecciona el modelo Ollama:",
            options=available_models,
            index=available_models.index(st.session_state.selected_model)
            if st.session_state.selected_model in available_models else 0
        )
        
        st.session_state.selected_model = selected_model
        
        st.markdown("---")
        
        # Control de temperatura
        temperature = st.slider(
            "Temperatura (creatividad del modelo):",
            min_value=0.0,
            max_value=1.0,
            value=0.7,
            step=0.1
        )
        
        st.markdown("---")
        
        # Información
        st.info(
            "📌 **Información**\n\n"
            "Este chatbot utiliza RAG para responder preguntas "
            "basadas en manuales de electrodomésticos.\n\n"
            "Asegúrate de tener Ollama ejecutándose localmente "
            "en http://localhost:11434"
        )
        
        return selected_model, temperature


def main():
    """Función principal de la aplicación Streamlit."""
    
    # Configuración de página
    st.set_page_config(
        page_title="Chatbot RAG - Manuales",
        page_icon="🤖",
        layout="wide"
    )
    
    # Inicializar sesión
    initialize_session_state()
    
    # Encabezado
    st.title("🤖 Asistente de Manuales de Electrodomésticos")
    st.markdown(
        "Haz preguntas específicas sobre los manuales y recibe respuestas "
        "precisas basadas en el contenido real."
    )
    
    # Configurar sidebar
    selected_model, temperature = setup_sidebar()
    
    # Sección principal
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("💬 Chat")
        
        # Inicializar chatbot si es necesario
        if st.session_state.chatbot is None:
            with st.spinner("🔄 Inicializando sistema RAG..."):
                try:
                    # Cargar datos
                    pdf_dir = "./manuales"  # Directorio con PDFs
                    if not os.path.exists(pdf_dir):
                        st.warning(
                            f"⚠️ Directorio '{pdf_dir}' no encontrado. "
                            "Por favor, crea el directorio y añade archivos PDF."
                        )
                    else:
                        vector_store, _ = initialize_rag_pipeline(pdf_dir)
                        st.session_state.vector_store = vector_store
                        
                        # Crear modelo
                        model_manager = OllamaModelManager()
                        llm = model_manager.create_llm(
                            selected_model, 
                            temperature=temperature
                        )
                        
                        # Crear chatbot
                        st.session_state.chatbot = RAGChatbot(vector_store, llm)
                        st.success("✓ Sistema RAG inicializado correctamente")
                
                except Exception as e:
                    st.error(f"Error al inicializar: {str(e)}")
        
        # Mostrar historial de chat
        for i, msg in enumerate(st.session_state.chat_history):
            if msg["role"] == "user":
                st.chat_message("user").write(msg["content"])
            else:
                st.chat_message("assistant").write(msg["content"])
        
        # Campo de entrada de preguntas
        user_question = st.chat_input(
            "Escribe tu pregunta sobre los manuales...",
            key="user_input"
        )
        
        if user_question and st.session_state.chatbot:
            # Añadir pregunta al historial
            st.session_state.chat_history.append({
                "role": "user",
                "content": user_question
            })
            
            # Mostrar pregunta
            st.chat_message("user").write(user_question)
            
            # Procesar respuesta
            with st.spinner("🔍 Buscando información..."):
                try:
                    response = st.session_state.chatbot.answer_question(
                        user_question
                    )
                    
                    # Mostrar respuesta
                    st.chat_message("assistant").write(response["answer"])
                    
                    # Mostrar fuentes
                    if response["sources"]:
                        with st.expander("📚 Fuentes consultadas"):
                            for source in set(response["sources"]):
                                st.caption(f"📄 {source}")
                    
                    # Guardar en historial
                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": response["answer"]
                    })
                    
                except Exception as e:
                    st.error(f"Error al procesar: {str(e)}")
    
    with col2:
        st.subheader("📊 Información")
        
        # Mostrar estado del sistema
        st.metric("Modelo activo", st.session_state.selected_model)
        st.metric("Temperatura", f"{temperature:.1f}")
        
        if st.session_state.chat_history:
            st.metric(
                "Mensajes en sesión",
                len(st.session_state.chat_history)
            )
        
        # Botón para limpiar historial
        if st.button("🗑️ Limpiar historial"):
            st.session_state.chat_history = []
            st.rerun()
        
        st.markdown("---")
        st.markdown(
            "**Modelo RAG:**\n"
            "- ChromaDB para almacenamiento vectorial\n"
            "- LangChain para orquestación\n"
            "- Ollama para inferencia local"
        )


if __name__ == "__main__":
    main()