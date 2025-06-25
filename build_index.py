# build_index.py (Versión v23 - Final Equilibrado y Eficiente)
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
from typing import List
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
EMBEDDING_MODEL = "text-embedding-3-large"
DOCS_FOLDER = "docs"
PERSIST_DIR = "chroma_db_v2" # Directorio para la nueva BBDD
VISION_TIMEOUT = 120
MAX_WORKERS = min(os.cpu_count() or 4, 4)
API_BASE = os.getenv("OPENAI_API_BASE", "https://genai-gateway.azure-api.net/")
API_KEY = os.getenv("OPENAI_API_KEY")

# --- Modelos y Splitter ---
vision_llm = ChatOpenAI(model=VISION_MODEL, openai_api_base=API_BASE, openai_api_key=API_KEY, max_tokens=4096, request_timeout=VISION_TIMEOUT)
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
    if not docs: return
    texts = [doc.page_content for doc in docs]
    metadatas = [doc.metadata for doc in docs]
    vector_store.add_texts(texts=texts, metadatas=metadatas)

def has_complex_layout(text: str, line_threshold: int = 15, short_line_chars: int = 60) -> bool:
    lines = text.splitlines()
    if not lines or len(lines) == 0: return False
    avg_line_len = sum(len(line) for line in lines) / len(lines)
    return len(lines) > line_threshold and avg_line_len < short_line_chars

# --- Procesadores de Ficheros ---

def process_pdf(path: str) -> List[Document]:
    basename = os.path.basename(path)
    doc_title = extract_document_title(basename)
    documents = []
    
    try:
        doc = fitz.open(path)
        for i, page in enumerate(tqdm(doc, desc=f"PDF: {basename}", leave=False), 1):
            meta = {"source": basename, "page": i, "title": doc_title}
            page_text = page.get_text("text").strip()

            if not page_text or has_complex_layout(page_text):
                logging.info(f"Página {i} de '{basename}' es compleja/imagen. Usando OCR de visión.")
                try:
                    pix = page.get_pixmap(dpi=200)
                    img_bytes = pix.tobytes("png")
                    vision_text = vision_extract(img_bytes)
                    if vision_text:
                        content = f"# Documento: {doc_title}\n## Página: {i}\n\n{vision_text}"
                        documents.append(Document(page_content=content, metadata=meta))
                except Exception as e:
                    logging.warning(f"Error procesando página {i} con OCR: {e}")
            else:
                content = f"# Documento: {doc_title}\n## Página: {i}\n\n{page_text}"
                documents.append(Document(page_content=content, metadata=meta))
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
            content = f"# Documento: {doc_title}\n\n{full_text}"
            documents.append(Document(page_content=content, metadata=meta))
    except Exception as e:
        logging.error(f"Error procesando '{basename}': {e}")
    return documents

def docx_extractor(path: str) -> str:
    doc = docx.Document(path)
    parts = [p.text for p in doc.paragraphs if p.text.strip()]
    for table in doc.tables:
        table_md = "\n".join([f"| {' | '.join(cell.text.strip() for cell in row.cells)} |" for row in table.rows])
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
    logging.warning(f"Extensión '{ext}' no soportada para el fichero {os.path.basename(path)}")
    return []

# --- Main Execution ---
if __name__ == "__main__":
    print(f"── Construyendo índice vectorial v23 (Final Equilibrado) en '{PERSIST_DIR}' ──")
    if os.path.exists(PERSIST_DIR):
        print(f"Borrando el directorio de índice antiguo: {PERSIST_DIR}")
        shutil.rmtree(PERSIST_DIR)
    
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
    print(f"✅ Índice vectorial creado con éxito y guardado en '{PERSIST_DIR}'.")