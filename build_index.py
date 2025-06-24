# build_index.py (Versión v21 - Calidad Equilibrada y Eficiente)
from __future__ import annotations
import os
import re
import glob
import json
import shutil
import base64
import signal
import logging
import fitz  # PyMuPDF
import pptx
import docx
import openpyxl
from concurrent.futures import ProcessPoolExecutor, as_completed
from io import BytesIO
from typing import List, Dict, Any
from tqdm import tqdm
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.schema.document import Document
from langchain.schema.messages import HumanMessage, SystemMessage
from langchain_chroma import Chroma
from langchain_experimental.text_splitter import SemanticChunker

# --- Configuración Global y Logging ---
load_dotenv()
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Constantes y Modelos ---
VISION_MODEL = "gpt-4o"
TEXT_MODEL = "gpt-4o"
EMBEDDING_MODEL = "text-embedding-3-large"
DOCS_FOLDER = "docs"
PERSIST_DIR = "chroma_db_v21" # Directorio para la nueva BBDD balanceada
VISION_TIMEOUT = 120
MAX_WORKERS = min(os.cpu_count() or 4, 4)
API_BASE = os.getenv("OPENAI_API_BASE", "https://genai-gateway.azure-api.net/")
API_KEY = os.getenv("OPENAI_API_KEY")

# --- Modelos y Splitter ---
vision_llm = ChatOpenAI(model=VISION_MODEL, openai_api_base=API_BASE, openai_api_key=API_KEY, max_tokens=4096, request_timeout=VISION_TIMEOUT)
text_llm = ChatOpenAI(model=TEXT_MODEL, openai_api_base=API_BASE, openai_api_key=API_KEY, max_tokens=2048)
embedder = OpenAIEmbeddings(model=EMBEDDING_MODEL, openai_api_base=API_BASE, openai_api_key=API_KEY, chunk_size=500)
text_splitter = SemanticChunker(embedder, breakpoint_threshold_type="percentile")

# --- Utilidades ---
def vision_extract(img_bytes: bytes) -> str:
    def handler(signum, frame): raise TimeoutError("La extracción de visión superó el tiempo límite.")
    signal.signal(signal.SIGALRM, handler)
    signal.alarm(VISION_TIMEOUT)
    try:
        b64_image = base64.b64encode(img_bytes).decode('utf-8')
        resp = vision_llm.invoke([
            SystemMessage(content="Eres un experto en OCR. Extrae todo el texto y la estructura de tablas de la imagen, formateando las tablas en Markdown limpio. Mantén la estructura original lo mejor posible."),
            HumanMessage(content=[
                {"type": "text", "text": "Extrae el texto y las tablas de esta imagen como Markdown."},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64_image}"}}
            ])
        ])
        return resp.content.strip()
    except Exception as e:
        logging.error(f"Error en la API de Visión: {e}")
        return ""
    finally:
        signal.alarm(0)

def extract_document_title(basename: str) -> str:
    title = os.path.splitext(basename)[0]
    title = re.sub(r'[\W_]+', ' ', title)
    title = re.sub(r'\s+', ' ', title).strip()
    return title.title()

def add_documents_to_vectorstore(docs: List[Document], vector_store: Chroma):
    if not docs:
        return
    texts = [doc.page_content for doc in docs]
    metadatas = [doc.metadata for doc in docs]
    vector_store.add_texts(texts=texts, metadatas=metadatas)

# --- NUEVA FUNCIÓN DE AYUDA para el enfoque Híbrido ---
def has_complex_layout(text: str, line_threshold: int = 10, short_line_chars: int = 50) -> bool:
    """Detecta si un texto parece tener un layout complejo (ej. tablas, listas)."""
    lines = text.splitlines()
    if not lines:
        return False
    # Heurística: si tiene muchas líneas y la longitud media de línea es corta, probablemente es complejo.
    avg_line_len = sum(len(line) for line in lines) / len(lines)
    return len(lines) > line_threshold and avg_line_len < short_line_chars


# --- Procesadores de Ficheros (Mejorados con Enfoque Híbrido) ---

def process_pdf(path: str) -> List[Document]:
    """Procesa PDFs con un enfoque HÍBRIDO: rápido para texto simple, visión para páginas complejas."""
    basename = os.path.basename(path)
    doc_title = extract_document_title(basename)
    documents = []
    
    try:
        doc = fitz.open(path)
        for i, page in enumerate(tqdm(doc, desc=f"PDF: {basename}", leave=False), 1):
            meta = {"source": basename, "page": i, "title": doc_title}
            
            # 1. Intenta la extracción de texto rápida
            page_text = page.get_text("text").strip()

            # 2. Decide si usar el modelo de visión
            # Condición: La página no tiene texto O parece tener un layout complejo (como una tabla)
            if not page_text or has_complex_layout(page_text):
                logging.info(f"Página {i} de '{basename}' es compleja/imagen. Usando OCR de visión.")
                try:
                    pix = page.get_pixmap(dpi=200)
                    img_bytes = pix.tobytes("png")
                    vision_text = vision_extract(img_bytes)
                    if vision_text:
                        content_with_header = f"# Documento: {doc_title}\n## Página: {i}\n\n{vision_text}"
                        documents.append(Document(page_content=content_with_header, metadata=meta))
                except Exception as e:
                    logging.warning(f"Error procesando página {i} de '{basename}' con OCR: {e}")
            else:
                # 3. Si es texto simple, úsalo directamente
                content_with_header = f"# Documento: {doc_title}\n## Página: {i}\n\n{page_text}"
                documents.append(Document(page_content=content_with_header, metadata=meta))
    except Exception as e:
        logging.error(f"Error crítico procesando PDF '{basename}': {e}")
    return documents

def process_generic(path: str, extractor: callable) -> List[Document]:
    basename = os.path.basename(path)
    doc_title = extract_document_title(basename)
    documents = []
    try:
        full_text = extractor(path)
        if full_text:
            meta = {"source": basename, "title": doc_title}
            content_with_header = f"# Documento: {doc_title}\n\n{full_text}"
            documents.append(Document(page_content=content_with_header, metadata=meta))
    except Exception as e:
        logging.error(f"Error procesando '{basename}': {e}")
    return documents

def docx_extractor(path: str) -> str:
    doc = docx.Document(path)
    parts = [p.text for p in doc.paragraphs if p.text.strip()]
    for table in doc.tables:
        table_md = "\n".join(["| " + " | ".join(cell.text.strip() for cell in row.cells) + " |" for row in table.rows])
        parts.append(f"\n--- TABLA ---\n{table_md}\n--- FIN TABLA ---\n")
    return "\n\n".join(parts)

def pptx_extractor(path: str) -> str:
    prs = pptx.Presentation(path)
    parts = []
    for i, slide in enumerate(prs.slides, 1):
        parts.append(f"\n## Diapositiva {i}\n")
        for shape in slide.shapes:
            if hasattr(shape, "text") and shape.text.strip():
                parts.append(shape.text)
    return "\n".join(parts)


def process_file(path: str) -> List[Document]:
    ext = path.lower().split('.')[-1]
    
    processor_map = {
        "pdf": process_pdf,
        "docx": lambda p: process_generic(p, docx_extractor),
        "pptx": lambda p: process_generic(p, pptx_extractor),
    }
    
    if ext in processor_map:
        return processor_map[ext](path)
    else:
        logging.warning(f"Extensión '{ext}' no soportada para el fichero {os.path.basename(path)}")
        return []


if __name__ == "__main__":
    print(f"── Construyendo índice vectorial v21 (Calidad Equilibrada) en '{PERSIST_DIR}' ──")
    if os.path.exists(PERSIST_DIR):
        print(f"Borrando el directorio de índice antiguo: {PERSIST_DIR}")
        shutil.rmtree(PERSIST_DIR)
    
    os.makedirs(PERSIST_DIR, exist_ok=True)
    db = Chroma(persist_directory=PERSIST_DIR, embedding_function=embedder)
    
    extensions = ("*.pdf", "*.pptx", "*.docx")
    files = [f for ext in extensions for f in glob.glob(os.path.join(DOCS_FOLDER, f"**/{ext}"), recursive=True)]
    
    print(f"Se encontraron {len(files)} documentos. Procesando con enfoque híbrido...")

    total_chunks = 0
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futs = {pool.submit(process_file, fp): fp for fp in files}
        for fut in tqdm(as_completed(futs), total=len(futs), desc="Procesando Ficheros"):
            try:
                raw_documents = fut.result()
                if raw_documents:
                    chunks = text_splitter.split_documents(raw_documents)
                    add_documents_to_vectorstore(chunks, db)
                    total_chunks += len(chunks)
            except Exception as e:
                logging.error(f"❌ Error crítico en el futuro del fichero {futs[fut]}: {e}", exc_info=True)

    print(f"\nSe generaron e indexaron un total de {total_chunks} chunks semánticos.")
    print(f"✅ Índice vectorial creado con éxito en '{PERSIST_DIR}'.")