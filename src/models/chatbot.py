# --- RAGChatbot.py ---
# (Versión modificada para usar el prompt externo)

from langchain.chains import RetrievalQA
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaLLM
# Importamos el prompt pre-configurado desde el nuevo archivo
from src.models.prompt_config import RAG_PROMPT


class RAGChatbot:
    """
    Implementa la lógica del chatbot con arquitectura RAG.
    Ahora importa el prompt desde prompt_config.py
    """
    
    def __init__(self, vector_store: Chroma, llm: OllamaLLM):
        """
        Inicializa el chatbot RAG.
        
        Args:
            vector_store: Base de datos vectorial con embeddings
            llm: Modelo de lenguaje de Ollama
        """
        self.vector_store = vector_store
        self.llm = llm
        self.retriever = vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 4}  # Recuperar los 4 fragmentos más relevantes
        )
        self.qa_chain = self._create_qa_chain()
    
    def _create_qa_chain(self) -> RetrievalQA:
        """
        Crea la cadena de preguntas y respuestas usando el prompt importado.
        
        Returns:
            Objeto RetrievalQA configurado
        """
        
        # Ya no definimos el template aquí.
        # Usamos directamente el objeto RAG_PROMPT importado de prompt_config.py
        
        qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.retriever,
            return_source_documents=True,
            chain_type_kwargs={"prompt": RAG_PROMPT}
        )
        
        return qa_chain
    
    def answer_question(self, question: str):
        """
        Procesa una pregunta y retorna la respuesta basada en RAG.
        
        Secuencia:
        1. Recibe la pregunta del usuario
        2. Busca fragmentos relevantes en la BD vectorial
        3. Formula prompt con contexto (usando RAG_PROMPT)
        4. Envía al modelo Ollama
        5. Retorna respuesta y fuentes
        
        Args:
            question: Pregunta del usuario
            
        Returns:
            Diccionario con respuesta y fuentes
        """
        result = self.qa_chain.invoke({"query": question})
        
        response_dict = {
            "answer": result["result"],
            "sources": [
                doc.metadata.get("source", "Unknown") 
                for doc in result["source_documents"]
            ]
        }
        
        return response_dict