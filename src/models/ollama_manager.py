from langchain_ollama import OllamaLLM
import os
import ollama

class OllamaModelManager:
    """
    Gestiona la conexión y selección de modelos de Ollama.
    """
    
    def __init__(self):
        """
        Inicializa el gestor de modelos Ollama leyendo la URL y clave del entorno.
        """
        self.base_url = os.getenv("OLLAMA_BASE_URL")
        # Lee la clave de API del entorno
        self.api_key = os.getenv("OLLAMA_API_KEY")
    
    def get_available_models(self) -> list:
        """
        Retorna la lista de modelos disponibles en el servidor de Ollama.
        """
        try:
            # Obtiene la lista de modelos del servidor Ollama
            models_info = ollama.list()
            # Extrae solo los nombres de los modelos
            return [model['name'] for model in models_info['models']]
        except Exception as e:
            print(f"⚠️  Advertencia: No se pudo conectar con el servidor de Ollama para obtener los modelos. {e}")
            return []
    
    def create_llm(self, model_name: str, temperature: float = 0.2) -> OllamaLLM:
        """
        Crea una instancia del modelo LLM seleccionado.
        
        Args:
            model_name: Nombre del modelo a utilizar
            temperature: Parámetro de creatividad del modelo (0-1)
            
        Returns:
            Instancia de OllamaLLM
        """
        # Configura los headers para la autenticación si se usa la nube
        headers = {}
        if self.api_key:
            headers['Authorization'] = f"Bearer {self.api_key}"
        
        llm = OllamaLLM(
            model=model_name,
            base_url=self.base_url,
            temperature=temperature,
            top_p=0.9,
            headers=headers # Pasa los headers al cliente de LangChain
        )
        return llm