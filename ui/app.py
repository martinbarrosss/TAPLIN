import streamlit as st
import os
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from dotenv import load_dotenv

# --- Importa las clases de tu proyecto ---
from src.rag.chatbot import RAGChatbot
from src.models.ollama_manager import OllamaModelManager

# --- Define la función de carga de la base de datos ---
def load_vector_store(persist_dir: str = "../chroma_db"):
    """Carga la base de datos vectorial persistente."""
    if not os.path.exists(persist_dir):
        return None
    
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    vector_store = Chroma(
        persist_directory=persist_dir,
        embedding_function=embeddings,
        collection_name="manuales_electrodomesticos"
    )
    return vector_store


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
        
        temperature = st.slider(
            "Temperatura (creatividad del modelo):",
            min_value=0.0,
            max_value=1.0,
            value=0.7,
            step=0.1
        )
        
        st.markdown("---")
        
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

    # Carga las variables del archivo .env
    load_dotenv()
    
    st.set_page_config(
        page_title="Chatbot RAG - Manuales",
        page_icon="🤖",
        layout="wide"
    )
    
    initialize_session_state()
    
    st.title("🤖 Asistente de Manuales de Electrodomésticos")
    st.markdown(
        "Haz preguntas específicas sobre los manuales y recibe respuestas "
        "precisas basadas en el contenido real."
    )
    
    selected_model, temperature = setup_sidebar()
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("💬 Chat")
        
        # Cargar el chatbot solo si no está en la sesión
        if st.session_state.chatbot is None:
            with st.spinner("🔄 Inicializando sistema RAG..."):
                try:
                    # Cargar la base de datos vectorial preexistente
                    vector_store = load_vector_store()
                    if vector_store is None:
                        st.error(
                            "❌ Base de datos vectorial no encontrada. "
                            "Por favor, ejecuta el script 'ingest_data.py' primero."
                        )
                    else:
                        st.session_state.vector_store = vector_store
                        
                        # Crear el modelo LLM y el chatbot
                        model_manager = OllamaModelManager()
                        llm = model_manager.create_llm(
                            selected_model, 
                            temperature=temperature
                        )
                        st.session_state.chatbot = RAGChatbot(vector_store, llm)
                        st.success("✓ Sistema RAG inicializado correctamente")
                
                except Exception as e:
                    st.error(f"Error al inicializar: {str(e)}")
        
        for i, msg in enumerate(st.session_state.chat_history):
            if msg["role"] == "user":
                st.chat_message("user").write(msg["content"])
            else:
                st.chat_message("assistant").write(msg["content"])
        
        user_question = st.chat_input(
            "Escribe tu pregunta sobre los manuales...",
            key="user_input"
        )
        
        if user_question and st.session_state.chatbot:
            st.session_state.chat_history.append({
                "role": "user",
                "content": user_question
            })
            
            st.chat_message("user").write(user_question)
            
            with st.spinner("🔍 Buscando información..."):
                try:
                    response = st.session_state.chatbot.answer_question(
                        user_question
                    )
                    
                    st.chat_message("assistant").write(response["answer"])
                    
                    if response["sources"]:
                        with st.expander("📚 Fuentes consultadas"):
                            for source in set(response["sources"]):
                                st.caption(f"📄 {source}")
                    
                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": response["answer"]
                    })
                    
                except Exception as e:
                    st.error(f"Error al procesar: {str(e)}")
    
    with col2:
        st.subheader("📊 Información")
        
        st.metric("Modelo activo", st.session_state.selected_model)
        st.metric("Temperatura", f"{temperature:.1f}")
        
        if st.session_state.chat_history:
            st.metric(
                "Mensajes en sesión",
                len(st.session_state.chat_history)
            )
        
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