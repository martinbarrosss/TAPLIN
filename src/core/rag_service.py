# src/core/rag_service.py

# Importaciones de dependencias (asumiendo que src es el root de tu proyecto):
from src.models.chatbot import RAGChatbot    
from src.models.traductor import Traductor    

class RAGMultilingualService:
    """
    Servicio central que orquesta la consulta RAG y la traducción (Gallego <-> Español).
    Este servicio es el que debe ser llamado por cualquier interfaz (consola, web, etc.).
    """
    def __init__(self, chatbot: RAGChatbot, traductor: Traductor):
        self.chatbot = chatbot
        self.traductor = traductor

    def answer_question(self, user_prompt: str, idioma_conversacion: str) -> dict:
        """
        Gestiona la lógica de traducción y la consulta RAG.
        
        Flujo (si es Gallego): Prompt(gl) -> Traducir(es) -> RAG -> Respuesta(es) -> Traducir(gl) -> Respuesta Final
        """
        
        prompt_a_rag = user_prompt
        
        # 1. TRADUCCIÓN DE IDA (gl -> es)
        if idioma_conversacion.lower() == 'gl':
            print(" [Service] Traducción GL->ES (IDA)...")
            prompt_a_rag = self.traductor.traducir_gl_a_es(user_prompt)
            
        # 2. CONSULTA RAG (siempre se hace en español)
        rag_response = self.chatbot.answer_question(prompt_a_rag)
        respuesta_rag_es = rag_response["answer"]
        
        final_response = respuesta_rag_es
        
        # 3. TRADUCCIÓN DE VUELTA (es -> gl)
        if idioma_conversacion.lower() == 'gl':
            print(" [Service] Traducción ES->GL (VUELTA)...")
            final_response = self.traductor.traducir_es_a_gl(respuesta_rag_es)
            
        # 4. Devolver la respuesta final y las fuentes
        return {
            "answer": final_response,
            "sources": rag_response["sources"]
        }