from langchain_ollama import OllamaLLM
import os

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
        
        self.available_models = [
            "gpt-oss:20b-cloud",
            "deepseek-v3.1:671b-cloud",
            "kimi-k2:1t-cloud",
            "qwen3-coder:480b-cloud"
        ]
    
    def get_available_models(self) -> list:
        """
        Retorna la lista de modelos disponibles.
        """
        return self.available_models
    
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