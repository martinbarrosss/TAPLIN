# DoBot - Asistente de Manuales de Electrodomésticos

DoBot es un chatbot asistente diseñado para responder preguntas sobre manuales de electrodomésticos en múltiples idiomas. Utiliza un enfoque de Generación Aumentada por Recuperación (RAG) para proporcionar respuestas precisas basadas en el contenido de los documentos proporcionados.

##  Características

- **Interfaz Dual:** Ejecuta el asistente a través de una interfaz web con Streamlit o una versión ligera enconsola de comandos.
- **Multilingüe:** Capaz de procesar y responder en diferentes idiomas.
- **Modelos Cloud:** Integrado con Ollama para utilizar modelos de lenguaje grandes (LLMs) sin depender del hardware propio.
- **Base de Datos Vectorial:** Usa ChromaDB para almacenar y consultar eficientemente la información extraída de los manuales.

##  Prerrequisitos

Antes de empezar, asegúrate de tener lo siguiente instalado en tu sistema:

- [Python 3.8+](https://www.python.org/downloads/)
- [Ollama](https://ollama.com/)

## ⚙️ Pasos de Configuración

Sigue estos pasos para configurar el entorno del proyecto.

### 1. Clona el Repositorio 
```sh
git clone https://github.com/martinbarrosss/TAPLIN.git
cd TAPLIN
```

### 2. Instala las Dependencias
Instala todos los paquetes de Python necesarios.
```sh
pip install -r requirements.txt
```

### 3. Añade los Manuales
Coloca los manuales de los electrodomésticos en formato `.pdf` dentro de la carpeta `manuals/`.

### 4. Procesa los Documentos
Ejecuta el siguiente script para procesar los manuales y crear la base de datos vectorial. Este paso solo es necesario una vez o cada vez que añadas nuevos documentos.
```sh
python src/data/ingest_data.py
```
Asegúrate de que Ollama esté en ejecución antes de lanzar este script.

##  Ejecutando la Aplicación

Una vez completada la configuración, puedes iniciar el asistente de dos maneras.

### Opción 1: Interfaz Web (Recomendado)
Para una experiencia gráfica completa que permite seleccionar el modelo, el idioma y otros parámetros:
```sh
streamlit run main.py
```

### Opción 2: Consola de Comandos
Para una interacción rápida y basada en texto:
```sh
python chat_consola.py
```

---
Ahora puedes empezar a chatear con DoBot y resolver tus dudas sobre los electrodomésticos.
