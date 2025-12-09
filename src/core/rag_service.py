# src/core/rag_service.py

from src.models.chatbot import RAGChatbot    
from src.models.traductor import Traductor    

class RAGMultilingualService:
    def __init__(self, chatbot: RAGChatbot, traductor: Traductor):
        self.chatbot = chatbot
        self.traductor = traductor
        # MEMORIA DE CORTO PLAZO
        # Guardará la pregunta técnica sugerida en el turno anterior
        self.last_suggestion = None 

    def _es_confirmacion(self, texto: str) -> bool:
        """Detecta si el usuario está diciendo simplemente 'sí' a la sugerencia."""
        palabras_afirmativas = ["si", "sí", "claro", "vale", "por favor", "ok", "yes"]
        texto_limpio = texto.lower().strip().replace(".", "").replace("!", "")
        return texto_limpio in palabras_afirmativas

    def answer_question(self, user_prompt: str, idioma_conversacion: str) -> dict:
        
        prompt_a_procesar = user_prompt
        usando_sugerencia = False

        # 1. TRADUCCIÓN DE IDA (gl -> es)
        if idioma_conversacion.lower() == 'gl':
            # Nota: Si el usuario dice "Si" en gallego, el traductor lo pasará a "Sí" en español
            prompt_a_procesar = self.traductor.traducir_gl_a_es(user_prompt)
        
        # 2. INTELIGENCIA DE FLUJO (Checkear memoria)
        # Si el usuario confirma y tenemos una sugerencia pendiente...
        if self._es_confirmacion(prompt_a_procesar) and self.last_suggestion:
            print(f" [Service] Usuario confirmó sugerencia. Usando: {self.last_suggestion}")
            prompt_a_procesar = self.last_suggestion
            usando_sugerencia = True
        
        # 3. CONSULTA RAG
        rag_response = self.chatbot.answer_question(prompt_a_procesar)
        respuesta_rag_es = rag_response["answer"]
        nueva_sugerencia = rag_response["suggested_question"]
        
        # Actualizamos la memoria con la NUEVA sugerencia para el siguiente turno
        self.last_suggestion = nueva_sugerencia
        
        # Formatear la respuesta final para incluir la pregunta visible
        final_text_es = respuesta_rag_es
        if nueva_sugerencia:
            # Añadimos el gancho visual al final de la respuesta
            final_text_es += f"\n\n_{nueva_sugerencia.lower()}_"

        # 4. TRADUCCIÓN DE VUELTA (es -> gl)
        final_response = final_text_es
        if idioma_conversacion.lower() == 'gl':
            final_response = self.traductor.traducir_es_a_gl(final_text_es)
            
        return {
            "answer": final_response,
            "sources": rag_response["sources"]
        }