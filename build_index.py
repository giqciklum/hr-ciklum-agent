# build_index.py  ─── Versión 2025-06-24 (Corrección de Despliegue)
"""
Reconstruye el índice vectorial utilizando una técnica de extracción multimodal y semántica.
Esta versión corrige un error en la búsqueda recursiva de ficheros que impedía
el despliegue en Cloud Build.
"""

from __future__ import annotations

import os
import re
import glob
import json
import shutil
import base64
import fitz  # PyMuPDF
import pptx
import docx
import openpyxl
from concurrent.futures import ProcessPoolExecutor, as_completed
from io import BytesIO
from typing import List

from tqdm import tqdm
from PIL import Image
from pdf2image import convert_from_path
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.schema.messages import HumanMessage, SystemMessage
from langchain_chroma import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter

# ── Configuración global ────────────────────────────────────────────────────
load_dotenv()
VISION_MODEL = "gpt-4o"
EMBEDDING_MODEL = "text-embedding-3-large"
DOCS_FOLDER = "docs"
CACHE_DIR = ".cache/doc_cache"
PERSIST_DIR = "chroma_db"
CHUNK_SIZE = 2048
CHUNK_OVERLAP = 100
MAX_WORKERS = min(os.cpu_count() or 4, 4)

API_BASE = "https://genai-gateway.azure-api.net/"
API_KEY = os.getenv("OPENAI_API_KEY")

# ── Modelos ────────────────────────────────────────────────────────────────
# Usamos una temperatura ligeramente superior para fomentar una generación de Q&A más rica.
llm = ChatOpenAI(model=VISION_MODEL, openai_api_base=API_BASE, openai_api_key=API_KEY, max_tokens=4096, temperature=0.1)
embedder = OpenAIEmbeddings(model=EMBEDDING_MODEL, openai_api_base=API_BASE, openai_api_key=API_KEY)
text_splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)

# ── Plantilla de Prompt Mejorada para la Extracción Semántica ────────────────
QA_SYSTEM_PROMPT = """
Eres un experto en extracción de conocimiento. Tu misión es analizar el contenido de un documento (proporcionado como texto y/o imágenes) y transformarlo en un formato de Preguntas y Respuestas (Q&A) exhaustivo.

**Instrucciones Clave:**
1.  **Análisis Integral:** Analiza todo el contenido, incluyendo texto, tablas, listas y la estructura del documento.
2.  **Perspectiva del Empleado:** Genera preguntas desde la perspectiva de un empleado de Ciklum. Piensa en todas las dudas que podrían surgir.
3.  **Exhaustividad:** No dejes ningún dato fuera. Cada política, procedimiento, nombre, rol, contacto, fecha límite o dato en una tabla debe convertirse en al menos un par de Q&A.
4.  **Respuestas Completas:** Las respuestas (R:) deben ser detalladas y contener toda la información relevante del documento para esa pregunta específica. Si una respuesta incluye un nombre, debe incluir también su rol si se menciona.
5.  **Formato de Salida Obligatorio:** Usa estrictamente este formato para cada par:

    P: [Pregunta clara y concisa del empleado]
    R: [Respuesta detallada y autocontenida]

**Ejemplo de una buena extracción:**
P: ¿Quién es el responsable de la prevención de riesgos laborales y qué debo hacer al respecto?
R: Julio Luis Jimenez Saenz es quien se encarga de la Salud en el trabajo y la Prevención de Riesgos Laborales. Al empezar, recibirás un correo de Preventiam para realizar una formación online de 120 minutos. También recibirás un correo de Docusign para firmar la documentación y aceptar o rechazar el examen médico voluntario.
"""

# ── Utilidades ─────────────────────────────────────────────────────────────
def extract_year(filename: str) -> int | None:
    m = re.compile(r"(20\d{2})").search(filename)
    return int(m.group(1)) if m else None

def generate_qa_from_content(content: List, filename: str) -> str:
    """Usa un LLM para convertir contenido (texto y/o imágenes) en Q&A."""
    if not content:
        return ""
    try:
        print(f"INFO: Generando Q&A para '{filename}'...")
        messages = [
            SystemMessage(content=QA_SYSTEM_PROMPT),
            HumanMessage(content=content)
        ]
        resp = llm.invoke(messages)
        return resp.content.strip()
    except Exception as e:
        print(f"⚠️  ERROR: No se pudo generar Q&A para '{filename}': {e}")
        return ""

# ── Extracción de Contenido por Tipo de Fichero ─────────────────────────────

def get_content_from_pdf(path: str) -> List:
    """Extrae contenido multimodal de cada página de un PDF."""
    content = [{"type": "text", "text": "Analiza el siguiente documento PDF y extrae su contenido en formato Q&A. El documento trata sobre el onboarding de Ciklum."}]
    try:
        images = convert_from_path(path)
        for i, image in enumerate(images):
            buffer = BytesIO()
            image.save(buffer, format="PNG")
            img_b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
            content.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_b64}"}})
            print(f"INFO: Procesada página {i+1} de '{os.path.basename(path)}' como imagen.")
    except Exception as e:
        print(f"⚠️  ADVERTENCIA: Falló la conversión de PDF a imagen para '{path}'. Se usará solo texto. Error: {e}")
        text = ""
        with fitz.open(path) as doc:
            for page in doc:
                text += page.get_text() + "\n\n"
        content.append({"type": "text", "text": text})
    return content

def get_content_from_pptx(path: str) -> List:
    """Extrae texto de una presentación de PowerPoint."""
    prs = pptx.Presentation(path)
    full_text = []
    for i, slide in enumerate(prs.slides):
        slide_title = f"--- Diapositiva {i+1} ---"
        slide_text_content = "\n".join(
            [shape.text for shape in slide.shapes if shape.has_text_frame]
        ).strip()
        if slide_text_content:
            full_text.append(f"{slide_title}\n{slide_text_content}")
    return [{"type": "text", "text": "\n\n".join(full_text)}]

def get_content_from_docx(path: str) -> List:
    """Extrae texto de un documento de Word."""
    doc = docx.Document(path)
    full_text = "\n".join([para.text for para in doc.paragraphs])
    return [{"type": "text", "text": full_text}]

def get_content_from_xlsx(path: str) -> List:
    """Extrae texto de todas las hojas de un fichero Excel, preservando la estructura."""
    workbook = openpyxl.load_workbook(path, data_only=True)
    full_text = []
    for sheetname in workbook.sheetnames:
        sheet = workbook[sheetname]
        table_markdown = f"### Hoja: {sheetname}\n\n"
        rows = list(sheet.iter_rows())
        if not rows:
            continue
        # Crear cabecera para markdown
        header = [str(cell.value) if cell.value is not None else "" for cell in rows[0]]
        table_markdown += "| " + " | ".join(header) + " |\n"
        table_markdown += "| " + " | ".join(["---"] * len(header)) + " |\n"
        # Añadir filas
        for row in rows[1:]:
            row_data = [str(cell.value) if cell.value is not None else "" for cell in row]
            table_markdown += "| " + " | ".join(row_data) + " |\n"
        full_text.append(table_markdown)
    return [{"type": "text", "text": "\n\n".join(full_text)}]

def get_content_from_image(path: str) -> List:
    """Prepara una imagen para el análisis de visión."""
    with open(path, "rb") as f:
        img_bytes = f.read()
    img_b64 = base64.b64encode(img_bytes).decode("utf-8")
    return [
        {"type": "text", "text": "Analiza esta imagen y extrae su contenido en formato Q&A."},
        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_b64}"}}
    ]

# ── Procesamiento de ficheros con la nueva lógica ───────────────────────────

def process_file(path: str) -> tuple[list[str], list[dict]]:
    basename = os.path.basename(path)
    cache_f = os.path.join(CACHE_DIR, basename + ".semantic_v2.json")
    
    if os.path.exists(cache_f) and os.path.getmtime(cache_f) > os.path.getmtime(path):
        with open(cache_f, encoding="utf-8") as f:
            cache = json.load(f)
        return cache["texts"], cache["metadatas"]

    ext = os.path.splitext(path)[1].lower()
    content_payload = []
    if ext == ".pdf":
        content_payload = get_content_from_pdf(path)
    elif ext == ".pptx":
        content_payload = get_content_from_pptx(path)
    elif ext == ".docx":
        content_payload = get_content_from_docx(path)
    elif ext == ".xlsx":
        content_payload = get_content_from_xlsx(path)
    elif ext in [".png", ".jpg", ".jpeg"]:
        content_payload = get_content_from_image(path)

    if not content_payload:
        return [], []

    qa_content = generate_qa_from_content(content_payload, basename)
    if not qa_content:
        return [], []
        
    chunks = text_splitter.split_text(qa_content)
    metadatas = [{"source": basename, "doc_year": extract_year(basename)} for _ in chunks]
    
    if chunks:
        with open(cache_f, "w", encoding="utf-8") as f:
            json.dump({"texts": chunks, "metadatas": metadatas}, f, ensure_ascii=False, indent=2)
            
    return chunks, metadatas

# ── Main ───────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("── Construyendo índice con Extracción Semántica Exhaustiva (Multimodal) ──")
    os.makedirs(CACHE_DIR, exist_ok=True)

    # --- BÚSQUEDA DE FICHEROS CORREGIDA ---
    # Se reemplaza la list comprehension con un bucle para mayor claridad y
    # para asegurar que glob.glob reciba los argumentos correctamente.
    extensions_to_process = ("*.pdf", "*.pptx", "*.docx", "*.xlsx", "*.png", "*.jpg", "*.jpeg")
    files = []
    for ext in extensions_to_process:
        # El patrón '**' busca en el directorio actual y todos los subdirectorios.
        pathname = os.path.join(DOCS_FOLDER, "**", ext)
        files.extend(glob.glob(pathname, recursive=True))
    
    print(f"Se encontraron {len(files)} documentos. Procesando con {MAX_WORKERS} workers…")
    # ----------------------------------------
    
    all_texts, all_metas = [], []
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futs = {pool.submit(process_file, fp): fp for fp in files}
        for fut in tqdm(as_completed(futs), total=len(futs), desc="Documentos"):
            fp = futs[fut]
            try:
                texts, metas = fut.result()
                all_texts.extend(texts)
                all_metas.extend(metas)
            except Exception as e:
                print(f"❌ ERROR FATAL procesando {fp}: {e}")

    if os.path.exists(PERSIST_DIR):
        print(f"INFO: Eliminando directorio de índice antiguo: {PERSIST_DIR}")
        shutil.rmtree(PERSIST_DIR)

    if all_texts:
        print(f"INFO: Creando nuevo índice vectorial con {len(all_texts)} chunks semánticos...")
        Chroma.from_texts(all_texts, embedder, metadatas=all_metas, persist_directory=PERSIST_DIR)
        print("✅ ¡Índice vectorial creado con éxito!")
    else:
        print("❌ ADVERTENCIA: No se generó contenido para indexar. Revisa los documentos y los logs.")
