# build_index.py  ─── Versión 2025-06-21 (Extracción Semántica Avanzada)
"""
Reconstruye el índice vectorial utilizando una técnica de extracción semántica.
En lugar de indexar texto plano, un LLM analiza cada documento y lo convierte
en un conjunto detallado de Preguntas y Respuestas (Q&A) para crear una base de
conocimiento más rica y contextual.
"""

from __future__ import annotations

import os
import re
import glob
import json
import shutil
import fitz  # PyMuPDF
import pptx
import docx # Para Word
import openpyxl # Para Excel
from concurrent.futures import ProcessPoolExecutor, as_completed
from io import BytesIO
from typing import List

from tqdm import tqdm
from PIL import Image
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.schema.messages import HumanMessage, SystemMessage
from langchain_chroma import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter

# ── Configuración global ────────────────────────────────────────────────────
load_dotenv()
QA_GENERATION_MODEL = "gpt-4o"
EMBEDDING_MODEL = "text-embedding-3-large"
DOCS_FOLDER = "docs"
CACHE_DIR = ".cache/doc_cache"
PERSIST_DIR = "chroma_db"
CHUNK_SIZE = 2048  # Ajustado para Q&A
CHUNK_OVERLAP = 100
MAX_WORKERS = min(os.cpu_count() or 4, 4)

API_BASE = "https://genai-gateway.azure-api.net/"
API_KEY = os.getenv("OPENAI_API_KEY")

# ── Modelos ────────────────────────────────────────────────────────────────
qa_llm = ChatOpenAI(model=QA_GENERATION_MODEL, openai_api_base=API_BASE, openai_api_key=API_KEY, max_tokens=4096, temperature=0.1)
embedder = OpenAIEmbeddings(model=EMBEDDING_MODEL, openai_api_base=API_BASE, openai_api_key=API_KEY)
text_splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)

# ── Plantilla de Prompt para la Extracción Semántica ────────────────────────
QA_SYSTEM_PROMPT = """
Eres un experto en gestión del conocimiento corporativo. Tu misión es leer el siguiente documento y transformarlo en un formato de Preguntas y Respuestas (Q&A) muy completo.
El objetivo es crear una base de conocimiento que un asistente de IA utilizará para responder a las preguntas de los empleados.

**Instrucciones:**
1.  **Analiza el Contenido:** Lee el texto completo del documento que te proporcionaré.
2.  **Identifica Conceptos Clave:** Extrae cada política, procedimiento, regla, beneficio, contacto, fecha límite, y cualquier otro dato importante.
3.  **Genera Pares de Q&A:** Para cada concepto clave, formula una pregunta clara y específica que un empleado podría hacer, y proporciona una respuesta detallada y completa basada únicamente en la información del texto.
4.  **Formato de Salida:** Presenta el resultado como una lista de preguntas y respuestas. Usa el siguiente formato para cada par:

    P: [Pregunta del empleado]
    R: [Respuesta detallada]

**Ejemplo:**
P: ¿Cuántos días de vacaciones anuales tengo?
R: En Ciklum, tienes derecho a 23 días laborables de vacaciones al año. Estos días deben ser utilizados antes del 31 de marzo del año siguiente al que se generaron.

Asegúrate de ser exhaustivo y de que cada respuesta contenga toda la información relevante del documento para esa pregunta específica.
"""

# ── Utilidades ─────────────────────────────────────────────────────────────
def extract_year(filename: str) -> int | None:
    m = re.compile(r"(20\d{2})").search(filename)
    return int(m.group(1)) if m else None

def generate_qa_from_text(text: str, filename: str) -> str:
    """Usa un LLM para convertir un bloque de texto en Q&A detallado."""
    if not text.strip():
        return ""
    try:
        print(f"Generando Q&A para {filename}...")
        resp = qa_llm.invoke([
            SystemMessage(content=QA_SYSTEM_PROMPT),
            HumanMessage(content=f"Aquí está el contenido del documento '{filename}':\n\n---\n\n{text}")
        ])
        return resp.content.strip()
    except Exception as e:
        print(f"⚠️ Error generando Q&A para {filename}: {e}")
        return ""

# ── Extracción de textos (simplificada para obtener el texto completo) ───────

def extract_text_from_pdf(path: str) -> str:
    """Extrae todo el texto de un PDF."""
    full_text = []
    with fitz.open(path) as doc:
        for page in doc:
            full_text.append(page.get_text("text").strip())
    return "\n\n".join(full_text)

def extract_text_from_pptx(path: str) -> str:
    """Extrae todo el texto de una presentación de PowerPoint."""
    prs = pptx.Presentation(path)
    full_text = []
    for slide in prs.slides:
        slide_text = "\n".join(
            [sh.text for sh in slide.shapes if getattr(sh, "text", "")]
        ).strip()
        if slide_text:
            full_text.append(slide_text)
    return "\n\n".join(full_text)

def extract_text_from_docx(path: str) -> str:
    """Extrae todo el texto de un documento de Word."""
    doc = docx.Document(path)
    return "\n".join([para.text for para in doc.paragraphs])

def extract_text_from_xlsx(path: str) -> str:
    """Extrae texto de todas las hojas de un fichero Excel."""
    workbook = openpyxl.load_workbook(path)
    full_text = []
    for sheetname in workbook.sheetnames:
        sheet = workbook[sheetname]
        sheet_text = "\n".join(
            [",".join([str(cell.value) for cell in row if cell.value is not None]) for row in sheet.iter_rows()]
        )
        if sheet_text:
            full_text.append(f"--- Hoja: {sheetname} ---\n{sheet_text}")
    return "\n\n".join(full_text)

def extract_text_from_image(path: str) -> str:
    """Usa GPT-4o Vision para extraer texto de una imagen."""
    try:
        with open(path, "rb") as f:
            img_bytes = f.read()
        b64_image = base64.b64encode(img_bytes).decode('utf-8')
        vision_llm = ChatOpenAI(model="gpt-4o", openai_api_base=API_BASE, openai_api_key=API_KEY, max_tokens=2048)
        resp = vision_llm.invoke([
            HumanMessage(
                content=[
                    {"type": "text", "text": "Extrae todo el texto de esta imagen en el formato original."},
                    {"type": "image_url", "image_url": f"data:image/jpeg;base64,{b64_image}"}
                ]
            )
        ])
        return resp.content.strip()
    except Exception as e:
        print(f"⚠️ Error procesando imagen {os.path.basename(path)}: {e}")
        return ""

# ── Procesamiento de ficheros con la nueva lógica ───────────────────────────

def process_file(path: str) -> tuple[list[str], list[dict]]:
    basename = os.path.basename(path)
    cache_f = os.path.join(CACHE_DIR, basename + ".semantic.json")
    
    # Comprobar si existe un caché válido
    if os.path.exists(cache_f) and os.path.getmtime(cache_f) > os.path.getmtime(path):
        with open(cache_f, encoding="utf-8") as f:
            cache = json.load(f)
        return cache["texts"], cache["metadatas"]

    # Extraer el texto en bruto del fichero
    ext = path.lower().split('.')[-1]
    raw_text = ""
    if ext == "pdf":
        raw_text = extract_text_from_pdf(path)
    elif ext == "pptx":
        raw_text = extract_text_from_pptx(path)
    elif ext == "docx":
        raw_text = extract_text_from_docx(path)
    elif ext == "xlsx":
        raw_text = extract_text_from_xlsx(path)
    elif ext in ["png", "jpg", "jpeg"]:
        raw_text = extract_text_from_image(path)

    if not raw_text.strip():
        return [], []

    # Generar el contenido semántico (Q&A)
    qa_content = generate_qa_from_text(raw_text, basename)
    if not qa_content:
        return [], []
        
    # Dividir el contenido Q&A en chunks y preparar metadatos
    chunks = text_splitter.split_text(qa_content)
    metadatas = [{"source": basename, "doc_year": extract_year(basename)} for _ in chunks]
    
    # Guardar el resultado en caché
    if chunks:
        with open(cache_f, "w", encoding="utf-8") as f:
            json.dump({"texts": chunks, "metadatas": metadatas}, f, ensure_ascii=False)
            
    return chunks, metadatas

# ── Main ───────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("── Construyendo índice vectorial con Extracción Semántica (Q&A) ──")
    os.makedirs(CACHE_DIR, exist_ok=True)

    extensions_to_process = ("*.pdf", "*.pptx", "*.docx", "*.xlsx", "*.png", "*.jpg", "*.jpeg")
    files = [f for ext in extensions_to_process for f in glob.glob(os.path.join(DOCS_FOLDER, ext))]
    print(f"Se encontraron {len(files)} documentos. Procesando con {MAX_WORKERS} workers…")
    
    all_texts, all_metas = [], []
    # Nota: El procesamiento con LLMs puede ser más lento. Se mantiene el pool para I/O.
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futs = {pool.submit(process_file, fp): fp for fp in files}
        for fut in tqdm(as_completed(futs), total=len(futs), desc="Documentos"):
            fp = futs[fut]
            try:
                texts, metas = fut.result()
                all_texts.extend(texts)
                all_metas.extend(metas)
            except Exception as e:
                print(f"❌ Error procesando {fp}: {e}")

    if os.path.exists(PERSIST_DIR):
        shutil.rmtree(PERSIST_DIR)

    if all_texts:
        Chroma.from_texts(all_texts, embedder, metadatas=all_metas, persist_directory=PERSIST_DIR)
        print(f"✅ Índice vectorial creado con {len(all_texts)} chunks semánticos.")
    else:
        print("❌ Sin contenido indexado. Comprueba los documentos o los logs de error.")
