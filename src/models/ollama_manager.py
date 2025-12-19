from langchain_ollama import OllamaLLM
import os

class OllamaModelManager:
    """
    Gestiona la conexión y selección de modelos de Ollama.
    """
    
    def __init__(self):
<<<<<<< HEAD
        # Configuración de la URL 

=======
        # Configuración de la URL
>>>>>>> 6bf16958d2be062256da8ea4ac7e545721bd1f73
        env_url = os.getenv("OLLAMA_BASE_URL", "")
        
        if "ollama.com" in env_url and not os.getenv("OLLAMA_API_KEY"):
            self.base_url = "http://127.0.0.1:11434"
        elif "localhost" in env_url:
            self.base_url = env_url.replace("localhost", "127.0.0.1")
        else:
            self.base_url = env_url if env_url else "http://127.0.0.1:11434"

        self.api_key = os.getenv("OLLAMA_API_KEY")
    
    def get_available_models(self) -> list:
        """
        Retorna una lista FIJA de modelos para evitar errores de conexión en Streamlit.
        """
<<<<<<< HEAD

=======
>>>>>>> 6bf16958d2be062256da8ea4ac7e545721bd1f73
        return [
            "gpt-oss:20b-cloud",
            "deepseek-v3.1:671b-cloud",
            "kimi-k2:1t-cloud",
            "qwen3-coder:480b-cloud"
        ]
    
    def create_llm(self, model_name: str, temperature: float = 0.2) -> OllamaLLM:
        """
        Crea una instancia del modelo LLM seleccionado.
        """
        headers = {}
        if self.api_key:
            headers['Authorization'] = f"Bearer {self.api_key}"
        
        llm = OllamaLLM(
            model=model_name,
            base_url=self.base_url,
            temperature=temperature,
            top_p=0.9,
            headers=headers
        )
        return llm