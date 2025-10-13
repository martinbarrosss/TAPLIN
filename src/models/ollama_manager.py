from langchain_ollama import OllamaLLM

class OllamaModelManager:
    """
    Gestiona la conexión y selección de modelos de Ollama.
    """
    
    def _init_(self, base_url: str = "http://localhost:11434"):
        """
        Inicializa el gestor de modelos Ollama.
        
        Args:
            base_url: URL base de la API de Ollama
        """
        self.base_url = base_url
        self.available_models = [
            "mistral",
            "llama2",
            "neural-chat",
            "orca-mini",
            "zephyr",
            "openchat"
        ]
    
    def get_available_models(self) -> List[str]:
        """
        Retorna la lista de modelos disponibles.
        
        Returns:
            Lista de nombres de modelos
        """
        return self.available_models
    
    def create_llm(self, model_name: str, temperature: float = 0.7) -> OllamaLLM:
        """
        Crea una instancia del modelo LLM seleccionado.
        
        Args:
            model_name: Nombre del modelo a utilizar
            temperature: Parámetro de creatividad del modelo (0-1)
            
        Returns:
            Instancia de OllamaLLM
        """
        llm = OllamaLLM(
            model=model_name,
            base_url=self.base_url,
            temperature=temperature,
            top_p=0.9,
        )
        return llm