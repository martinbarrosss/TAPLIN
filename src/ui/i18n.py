# src/ui/i18n.py
# NUEVO ARCHIVO

TRANSLATIONS = {
    "es": {
        # main.py
        "page_title": "Chatbot RAG - Manuales",
        "header_title": "Asistente de Manuales de Electrodomésticos",
        "header_markdown": "Haz preguntas específicas sobre los manuales y recibe respuestas precisas basadas en el contenido real.",
        "tab_chat": "💬 Chat",
        "tab_info": "ℹ️ Información",
        "tab_history": "📜 Historial",
        
        # app_ui.py (sidebar)
        "sidebar_title": " Configuración",
        "sidebar_model_select": "Selecciona el modelo Ollama:",
        "sidebar_lang_select": "Idioma de la conversación:",
        "sidebar_lang_es": "Español (es)",
        "sidebar_lang_gl": "Gallego (gl)",
        "sidebar_temp_slider": "Temperatura (creatividad del modelo):",
        "sidebar_info_title": "Información",
        "sidebar_info_body": (
            "Este chatbot utiliza RAG para responder preguntas "
            "basadas en manuales de electrodomésticos."
        ),
        
        # renders.py (tab_chat)
        "chat_subheader": "Chat en {lang}",
        "chat_input_placeholder": "Escribe tu pregunta sobre los manuales en {lang}...",
        "chat_spinner": " ⚙️ Procesando consulta y traduciendo a {lang}...",
        "chat_expander_sources": " Fuentes consultadas",
        "chat_error": "Error al procesar: {error}",
        
        # renders.py (tab_info)
        "info_subheader": "Información del Sistema",
        "info_metric_model": "Modelo activo",
        "info_metric_temp": "Temperatura",
        "info_metric_lang": "Idioma de la sesión",
        "info_metric_messages": "Mensajes en sesión",
        "info_rag_model_title": "**Modelo RAG:**",
        "info_rag_model_body": (
            "- ChromaDB para almacenamiento vectorial\n"
            "- LangChain para orquestación\n"
            "- Ollama para inferencia (Local/Cloud)\n"
            "- **Traductor:** Helsinki-NLP (GL <-> ES)"
        ),
        
        # renders.py (tab_history)
        "history_subheader": "Revisar Historial de la Sesión",
        "history_button_clear": "🗑️ Limpiar historial",
        "history_caption_empty": "El historial de esta sesión está vacío.",
        "history_role_user": "**Tú:** {content}",
        "history_role_assistant": "**Asistente:** {content}",
    },
    "gl": {
        # main.py
        "page_title": "Chatbot RAG - Manuais",
        "header_title": "Asistente de Manuais de Electrodomésticos",
        "header_markdown": "Fai preguntas específicas sobre os manuais e recibe respostas precisas baseadas no contido real.",
        "tab_chat": "💬 Chat",
        "tab_info": "ℹ️ Información",
        "tab_history": "📜 Historial",
        
        # app_ui.py (sidebar)
        "sidebar_title": " Configuración",
        "sidebar_model_select": "Selecciona o modelo Ollama:",
        "sidebar_lang_select": "Idioma da conversación:",
        "sidebar_lang_es": "Español (es)",
        "sidebar_lang_gl": "Galego (gl)",
        "sidebar_temp_slider": "Temperatura (creatividade do modelo):",
        "sidebar_info_title": "Información",
        "sidebar_info_body": (
            "Este chatbot utiliza RAG para responder preguntas "
            "baseadas en manuais de electrodomésticos."
        ),
        
        # renders.py (tab_chat)
        "chat_subheader": "Chat en {lang}",
        "chat_input_placeholder": "Escribe a túa pregunta sobre os manuais en {lang}...",
        "chat_spinner": " ⚙️ Procesando consulta e traducindo a {lang}...",
        "chat_expander_sources": " Fontes consultadas",
        "chat_error": "Erro ao procesar: {error}",
        
        # renders.py (tab_info)
        "info_subheader": "Información do Sistema",
        "info_metric_model": "Modelo activo",
        "info_metric_temp": "Temperatura",
        "info_metric_lang": "Idioma da sesión",
        "info_metric_messages": "Mensaxes en sesión",
        "info_rag_model_title": "**Modelo RAG:**",
        "info_rag_model_body": (
            "- ChromaDB para almacenamento vectorial\n"
            "- LangChain para orquestración\n"
            "- Ollama para inferencia (Local/Cloud)\n"
            "- **Tradutor:** Helsinki-NLP (GL <-> ES)"
        ),
        
        # renders.py (tab_history)
        "history_subheader": "Revisar Historial da Sesión",
        "history_button_clear": "🗑️ Limpar historial",
        "history_caption_empty": "O historial desta sesión está baleiro.",
        "history_role_user": "**Ti:** {content}",
        "history_role_assistant": "**Asistente:** {content}",
    }
}

def get_translator(language_code: str):
    """
    Retorna una función 't' que traduce claves para el idioma dado.
    """
    # Si el idioma no existe, usa español como fallback
    lang = language_code if language_code in TRANSLATIONS else "es"
    
    def t(key: str, **kwargs) -> str:
        """Busca la clave en el diccionario de traducciones."""
        translation = TRANSLATIONS[lang].get(key, f"_{key}_") # Devuelve la clave si no se encuentra
        if kwargs:
            try:
                return translation.format(**kwargs)
            except KeyError:
                return translation # Evita errores si faltan kwargs
        return translation
    
    return t