import os
import glob
from pathlib import Path
from typing import List, Dict

import pymupdf
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document

class PDFProcessor:
    """
    Clase para procesar archivos PDF y preparar los datos para RAG.
    """
    
    def __init__(self, pdf_directory: str, chunk_size: int = 800, 
                 chunk_overlap: int = 200):
        """
        Inicializa el procesador de PDFs.
        
        Args:
            pdf_directory: Ruta del directorio con archivos PDF
            chunk_size: Tamaño de cada fragmento de texto
            chunk_overlap: Solapamiento entre fragmentos consecutivos
        """
        self.pdf_directory = pdf_directory
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.documents = []
        
    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """
        Extrae el texto completo de un archivo PDF.
        
        Args:
            pdf_path: Ruta del archivo PDF
            
        Returns:
            Texto extraído del PDF
        """
        text = ""
        try:
            pdf_document = pymupdf.open(pdf_path)
            for page_num in range(len(pdf_document)):
                page = pdf_document[page_num]
                text += page.get_text()
            pdf_document.close()
            print(f"✓ Texto extraído de: {os.path.basename(pdf_path)}")
        except Exception as e:
            print(f"✗ Error al procesar {pdf_path}: {str(e)}")
        return text
    
    def load_all_pdfs(self) -> List[Document]:
        """
        Carga todos los PDFs del directorio especificado.
        
        Returns:
            Lista de documentos LangChain
        """
        pdf_files = glob.glob(os.path.join(self.pdf_directory, "*.pdf"))
        
        if not pdf_files:
            print(f"⚠ No se encontraron archivos PDF en: {self.pdf_directory}")
            return []
        
        print(f"\n📄 Procesando {len(pdf_files)} archivo(s) PDF...")
        
        for pdf_file in pdf_files:
            text = self.extract_text_from_pdf(pdf_file)
            if text:
                # Crear documento con metadatos
                doc = Document(
                    page_content=text,
                    metadata={"source": os.path.basename(pdf_file)}
                )
                self.documents.append(doc)
        
        return self.documents
    
    def split_documents(self, documents: List[Document]) -> List[Document]:
        """
        Divide los documentos en fragmentos más pequeños (chunks).
        
        Args:
            documents: Lista de documentos a dividir
            
        Returns:
            Lista de documentos divididos
        """
        print(f"\n✂️ Dividiendo documentos en chunks...")
        
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        
        split_docs = splitter.split_documents(documents)
        print(f"✓ {len(split_docs)} chunks creados")
        
        return split_docs
    
    def create_vector_store(self, documents: List[Document], 
                           persist_dir: str = "./chroma_db") -> Chroma:
        """
        Crea y almacena los embeddings en ChromaDB.
        
        Args:
            documents: Lista de documentos a embeber
            persist_dir: Directorio para persistencia de la BD vectorial
            
        Returns:
            Objeto Chroma con la base de datos vectorial
        """
        print(f"\n🔧 Generando embeddings e inicializando ChromaDB...")
        
        # Usar embeddings de HuggingFace (modelo lightweight)
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        
        # Crear o cargar base de datos vectorial
        vector_store = Chroma.from_documents(
            documents=documents,
            embedding=embeddings,
            persist_directory=persist_dir,
            collection_name="manuales_electrodomesticos"
        )
        
        print(f"✓ Vector store creado en: {persist_dir}")
        return vector_store


def initialize_rag_pipeline(pdf_directory: str) -> tuple:
    """
    Función auxiliar para inicializar todo el pipeline de RAG.
    
    Args:
        pdf_directory: Ruta del directorio con PDFs
        
    Returns:
        Tupla (vector_store, embeddings)
    """
    processor = PDFProcessor(pdf_directory)
    documents = processor.load_all_pdfs()
    split_docs = processor.split_documents(documents)
    vector_store = processor.create_vector_store(split_docs)
    
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    
    return vector_store, embeddings