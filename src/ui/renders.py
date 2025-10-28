# renders.py
import streamlit as st

# --- Funciones de Renderizado (Pestañas) ---

def render_tab_chat():
    """Lógica y renderizado de la pestaña de Chat (manejo de la conversación)."""
    st.subheader("Chat")
    
    # Historial de chat (visualización principal)
    for msg in st.session_state.chat_history:
        st.chat_message(msg["role"]).write(msg["content"])
    
    user_question = st.chat_input("Escribe tu pregunta sobre los manuales...", key="user_input")
    
    # Bloque de Lógica de Respuesta del Chat
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

def render_tab_info():
    """Lógica y renderizado de la pestaña de Información del Sistema."""
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

def render_tab_history():
    """Lógica y renderizado de la pestaña de Historial de la Sesión."""
    st.subheader("Revisar Historial de la Sesión")

    if st.button("🗑️ Limpiar historial"):
        st.session_state.chat_history = []
        st.rerun()
    
    st.markdown("---")

    if not st.session_state.chat_history:
        st.caption("El historial de esta sesión está vacío.")
    else:
        for msg in st.session_state.chat_history:
            if msg["role"] == "user":
                st.markdown(f"**Tú:** {msg['content']}")
            else:
                st.markdown(f"**Asistente:** {msg['content']}")
            st.markdown("---") # Separador