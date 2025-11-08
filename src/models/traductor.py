# src/models/traductor.py

from transformers import pipeline

class Traductor:
    """Clase que maneja los pipelines de traducción Gallego <-> Español (GL <-> ES)."""
    
    def __init__(self):
        # Modelos de traducción Opus-MT de Helsinki-NLP
        MODELO_GL_ES = "Helsinki-NLP/opus-mt-gl-es"
        MODELO_ES_GL = "Helsinki-NLP/opus-mt-es-gl"
        
        # Inicialización de pipelines
        print(f"Cargando el traductor GL->ES: {MODELO_GL_ES}...")
        self.traductor_gl_es = pipeline("translation", model=MODELO_GL_ES)
        print("Traductor GL->ES cargado.")

        print(f"Cargando el traductor ES->GL: {MODELO_ES_GL}...")
        self.traductor_es_gl = pipeline("translation", model=MODELO_ES_GL)
        print("Traductor ES->GL cargado.")

    def traducir_gl_a_es(self, texto_gallego: str) -> str:
        """Traduce de Gallego a Español."""
        if not texto_gallego.strip():
            return ""
        resultado = self.traductor_gl_es(texto_gallego)[0]
        return resultado['translation_text']

    def traducir_es_a_gl(self, texto_espanol: str) -> str:
        """Traduce de Español a Gallego."""
        if not texto_espanol.strip():
            return ""
        resultado = self.traductor_es_gl(texto_espanol)[0]
        return resultado['translation_text']