# src/ui/renders.py

import streamlit as st
import streamlit.components.v1 as components

# --- Funciones de Renderizado (Pestañas) ---

def render_tab_chat(t):
    """Lógica y renderizado de la pestaña de Chat (manejo de la conversación)."""
    
    # Determinamos el idioma de la conversación
    idioma = st.session_state.language 
    
    # Etiqueta para el usuario
    st.subheader(t("chat_subheader", lang=idioma.upper()))
    
    # --- Historial de chat (visualización principal) ---
    history = st.session_state.chat_history
    
    for i, msg in enumerate(history): 
        st.chat_message(msg["role"]).write(msg["content"])
        
        # Mostrar fuentes solo en el mensaje MÁS RECIENTE del asistente
        if (i == len(history) - 1 and 
            msg["role"] == "assistant" and 
            st.session_state.get("last_sources")):
            
            with st.expander(t("chat_expander_sources")): 
                # Extraemos solo los nombres de archivo de los objetos Document
                unique_sources = set()
                for doc in st.session_state.last_sources:
                    # doc.metadata es un diccionario. Buscamos la clave "source"
                    if hasattr(doc, 'metadata'):
                        unique_sources.add(doc.metadata.get("source", "Desconocido"))
                    else:
                        unique_sources.add(str(doc)) # Fallback por seguridad

                for source_name in unique_sources:
                    st.caption(f"📄 {source_name}")
            
            st.session_state.last_sources = [] # Limpiamos


    # --- Input en la parte inferior (estándar) ---
    user_question = st.chat_input(
        t("chat_input_placeholder", lang=idioma.upper()), 
        key="user_input"
    )

    # --- Lógica de Respuesta ---
    if user_question and st.session_state.chatbot: 
        st.session_state.chat_history.append({"role": "user", "content": user_question})
        
        response = None
        with st.spinner(t("chat_spinner", lang=idioma.upper())):
            try:
                response = st.session_state.chatbot.answer_question(
                    user_question, 
                    idioma
                )
            except Exception as e:
                st.error(t("chat_error", error=str(e))) 

        if response:
            st.session_state.chat_history.append({"role": "assistant", "content": response["answer"]})
            # Guardamos los objetos Document completos en el estado
            st.session_state.last_sources = response["sources"]
        else:
            st.session_state.last_sources = []
        
        # El st.rerun() sigue siendo necesario para refrescar
        st.rerun()
    
    # --- SCRIPT DE AUTO-SCROLL ---
    components.html(
        """
        <script>
            setTimeout(function(){
                window.scrollTo(0, document.body.scrollHeight);
            }, 50);
        </script>
        """,
        height=0, 
    )


def render_tab_info(t):
    """Lógica y renderizado de la pestaña de Información del Sistema."""
    st.subheader(t("info_subheader")) 
    
    st.metric(t("info_metric_model"), st.session_state.selected_model) 
    st.metric(t("info_metric_temp"), f"{st.session_state.temperature:.1f}") 
    st.metric(t("info_metric_lang"), st.session_state.language.upper()) 
    
    if st.session_state.chat_history:
        st.metric(t("info_metric_messages"), len(st.session_state.chat_history)) 

    st.markdown("---")
    st.markdown(
        f"{t('info_rag_model_title')}\n" 
        f"{t('info_rag_model_body')}" 
    )

def render_tab_history(t):
    """Lógica y renderizado de la pestaña de Historial de la Sesión."""
    st.subheader(t("history_subheader")) 

    if st.button(t("history_button_clear")): 
        st.session_state.chat_history = []
        st.session_state.last_sources = [] 
        st.rerun()
    
    st.markdown("---")

    if not st.session_state.chat_history:
        st.caption(t("history_caption_empty")) 
    else:
        for msg in st.session_state.chat_history:
            if msg["role"] == "user":
                st.markdown(t("history_role_user", content=msg['content'])) 
            else:
                st.markdown(t("history_role_assistant", content=msg['content'])) 
            st.markdown("---")