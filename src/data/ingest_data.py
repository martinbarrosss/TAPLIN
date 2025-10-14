import os
import glob
from pathlib import Path
from typing import List, Dict

import pymupdf
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document


class PDFProcessor:
    """Clase para procesar archivos PDF y preparar los datos para RAG."""
    def __init__(self, pdf_directory: str, chunk_size: int = 800, 
                 chunk_overlap: int = 200):
        self.pdf_directory = pdf_directory
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.documents = []
        
    def extract_text_from_pdf(self, pdf_path: str) -> str:
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
        pdf_files = glob.glob(os.path.join(self.pdf_directory, "*.pdf"))
        if not pdf_files:
            print(f"⚠ No se encontraron archivos PDF en: {self.pdf_directory}")
            return []
        print(f"\n📄 Procesando {len(pdf_files)} archivo(s) PDF...")
        for pdf_file in pdf_files:
            text = self.extract_text_from_pdf(pdf_file)
            if text:
                doc = Document(
                    page_content=text,
                    metadata={"source": os.path.basename(pdf_file)}
                )
                self.documents.append(doc)
        return self.documents
    
    def split_documents(self, documents: List[Document]) -> List[Document]:
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
        print(f"\n🔧 Generando embeddings e inicializando ChromaDB...")
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        vector_store = Chroma.from_documents(
            documents=documents,
            embedding=embeddings,
            persist_directory=persist_dir,
            collection_name="manuales_electrodomesticos"
        )
        print(f"✓ Vector store creado en: {persist_dir}")
        return vector_store

# --- Lógica principal del script de ingesta ---
def ingest_data(pdf_dir: Path):
    """Ejecuta el pipeline de ingesta de datos."""

    if not os.path.exists(pdf_dir):
        print(f"❌ Error: El directorio '{pdf_dir}' no existe. Por favor, créalo y añade tus archivos PDF.")
        return
        
    processor = PDFProcessor(pdf_dir)
    documents = processor.load_all_pdfs()
    if documents:
        split_docs = processor.split_documents(documents)
        processor.create_vector_store(split_docs, persist_dir="./chroma_db")
    
if __name__ == "__main__":

    pdf_dir = Path("./manuals")
    persist_dir = Path("./chroma_db")
    ingest_data(pdf_dir)