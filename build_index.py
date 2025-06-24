# build_index.py (Versión Final y Corregida)
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
from PIL import Image
from pdf2image import convert_from_path
from dotenv import load_dotenv
from transformers import AutoTokenizer
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.schema.messages import HumanMessage, SystemMessage
from langchain_chroma import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter

# --- Configuración Global y Logging ---
load_dotenv()
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Constantes y Modelos ---
VISION_MODEL = "gpt-4o"
TEXT_MODEL = "gpt-4o"
EMBEDDING_MODEL = "text-embedding-3-large"
DOCS_FOLDER = "docs"
CACHE_DIR = ".cache/doc_cache"
PERSIST_DIR = "chroma_db_v2" # Unificado con app.py
CHUNK_SIZE = 800
CHUNK_OVERLAP = 100
MAX_WORKERS = min(os.cpu_count() or 4, 4)
VISION_TIMEOUT = 90
API_BASE = os.getenv("OPENAI_API_BASE", "https://genai-gateway.azure-api.net/")
API_KEY = os.getenv("OPENAI_API_KEY")

# --- Modelos y Splitter ---
vision_llm = ChatOpenAI(model=VISION_MODEL, openai_api_base=API_BASE, openai_api_key=API_KEY, max_tokens=4096, request_timeout=VISION_TIMEOUT)
text_llm = ChatOpenAI(model=TEXT_MODEL, openai_api_base=API_BASE, openai_api_key=API_KEY, max_tokens=2048)
embedder = OpenAIEmbeddings(model=EMBEDDING_MODEL, openai_api_base=API_BASE, openai_api_key=API_KEY)
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
text_splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
    tokenizer,
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    separators=["\n\n", "\n", ". ", " ", ""],
)

# --- Utilidades ---
YEAR_PAT = re.compile(r"(20\d{2})")
BOILERPLATE_PAT = re.compile(r"^(página \d+|\d+ \| © \d{4} ciklum|confidencial|documento interno).*$", re.IGNORECASE | re.MULTILINE)

def extract_year(filename: str) -> int | None:
    m = YEAR_PAT.search(filename)
    return int(m.group(1)) if m else None

def generate_questions(chunk: str) -> str:
    try:
        resp = text_llm.invoke([
            SystemMessage(content="Genera 3 preguntas complejas que un empleado podría hacer y cuya respuesta esté en el siguiente fragmento de un documento interno. Sé conciso."),
            HumanMessage(content=chunk)
        ])
        return resp.content.strip()
    except Exception as e:
        logging.warning(f"No se pudieron generar preguntas hipotéticas: {e}")
        return ""

def add_chunk(text_content: str, meta: dict, texts: list[str], metas: list[dict]):
    cleaned_content = BOILERPLATE_PAT.sub("", text_content).strip()
    if len(cleaned_content) < 20: # Ignorar fragmentos muy pequeños
        return
    chunks = text_splitter.split_text(cleaned_content)
    for chunk in chunks:
        # Enriquecer el chunk con preguntas hipotéticas para mejorar la recuperación (HyDE)
        enriched_chunk = f"PREGUNTAS HIPOTÉTICAS QUE ESTE TEXTO PODRÍA RESPONDER:\n{generate_questions(chunk)}\n---\nCONTENIDO DEL DOCUMENTO:\n{chunk}"
        texts.append(enriched_chunk)
        metas.append(meta.copy())

def vision_extract(img_bytes: bytes) -> str:
    # Esta función de timeout es para sistemas UNIX-like, como el contenedor Docker.
    def handler(signum, frame): raise TimeoutError("La extracción de texto de la imagen ha superado el tiempo límite.")
    signal.signal(signal.SIGALRM, handler)
    signal.alarm(VISION_TIMEOUT)
    try:
        b64_image = base64.b64encode(img_bytes).decode('utf-8')
        resp = vision_llm.invoke([
            SystemMessage(content="Eres un experto en OCR. Extrae todo el texto y la estructura de tablas de la imagen, formateando las tablas en Markdown. Ignora texto irrelevante o artefactos de la imagen."),
            HumanMessage(content=[
                {"type": "text", "text": "Extrae el texto de esta imagen."},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64_image}"}}
            ])
        ])
        return resp.content.strip()
    except Exception as e:
        logging.error(f"Error en la API de Visión: {e}")
        return ""
    finally:
        signal.alarm(0) # Desactivar la alarma

# --- Procesadores de Ficheros (Mejorados) ---

def process_pdf(path: str) -> tuple[list[str], list[dict]]:
    texts, metas = [], []
    basename = os.path.basename(path)
    year = extract_year(basename)
    try:
        with fitz.open(path) as doc:
            for i, page in enumerate(doc, 1):
                meta = {"source": basename, "page": i, "doc_year": year}
                page_text = page.get_text("text").strip()
                if page_text:
                    add_chunk(page_text, meta, texts, metas)
                else: # Página es una imagen
                    logging.info(f"Página {i} de '{basename}' no tiene texto, intentando OCR...")
                    try:
                        images = convert_from_path(path, first_page=i, last_page=i, dpi=200)
                        if images:
                            buf = BytesIO()
                            images[0].save(buf, format="PNG")
                            vision_text = vision_extract(buf.getvalue())
                            if vision_text:
                                add_chunk(vision_text, meta, texts, metas)
                    except Exception as e:
                        logging.warning(f"Error en OCR para pág {i} de '{basename}': {e}")
    except Exception as e:
        logging.error(f"Error crítico procesando PDF '{basename}': {e}")
    return texts, metas

def process_docx(path: str) -> tuple[list[str], list[dict]]:
    texts, metas = [], []
    basename = os.path.basename(path)
    year = extract_year(basename)
    try:
        doc = docx.Document(path)
        full_text_parts = [p.text for p in doc.paragraphs if p.text.strip()]
        
        # [MEJORA] Extraer texto de tablas y convertir a Markdown
        for table in doc.tables:
            table_md = "\n".join(["| " + " | ".join(cell.text.strip() for cell in row.cells) + " |" for row in table.rows])
            full_text_parts.append(f"\n--- INICIO TABLA ---\n{table_md}\n--- FIN TABLA ---\n")

        full_text = "\n\n".join(full_text_parts).strip()
        if full_text:
            add_chunk(full_text, {"source": basename, "doc_year": year}, texts, metas)
    except Exception as e:
        logging.error(f"Error procesando DOCX '{basename}': {e}")
    return texts, metas

def process_pptx(path: str) -> tuple[list[str], list[dict]]:
    texts, metas = [], []
    basename = os.path.basename(path)
    year = extract_year(basename)
    try:
        prs = pptx.Presentation(path)
        for i, slide in enumerate(prs.slides, 1):
            meta = {"source": basename, "slide": i, "doc_year": year}
            slide_text_parts = []
            for shape in slide.shapes:
                if hasattr(shape, "text") and shape.text.strip():
                    slide_text_parts.append(shape.text)
                
                # [MEJORA] Extraer texto de imágenes en la diapositiva
                if shape.shape_type == pptx.enum.shapes.MSO_SHAPE_TYPE.PICTURE:
                    try:
                        image_bytes = shape.image.blob
                        vision_text = vision_extract(image_bytes)
                        if vision_text:
                            slide_text_parts.append(f"\n--- IMAGEN EXTRAÍDA ---\n{vision_text}\n--- FIN IMAGEN ---")
                    except Exception as e:
                        logging.warning(f"No se pudo procesar imagen en slide {i} de '{basename}': {e}")
            
            slide_text = "\n\n".join(slide_text_parts).strip()
            if slide_text:
                add_chunk(slide_text, meta, texts, metas)
    except Exception as e:
        logging.error(f"Error procesando PPTX '{basename}': {e}")
    return texts, metas

def process_xlsx(path: str) -> tuple[list[str], list[dict]]:
    texts, metas = [], []
    basename = os.path.basename(path)
    year = extract_year(basename)
    try:
        wb = openpyxl.load_workbook(path, data_only=True)
        for sheetname in wb.sheetnames:
            meta = {"source": basename, "sheet": sheetname, "doc_year": year}
            sheet = wb[sheetname]
            rows = list(sheet.iter_rows())
            if not rows: continue

            # [MEJORA] Convertir hoja a tabla Markdown para preservar la estructura
            try:
                header = [str(cell.value or "").strip() for cell in rows[0]]
                md_lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
                for row in rows[1:]:
                    values = [str(cell.value or "").strip() for cell in row]
                    if any(values): # Solo añadir filas que no estén vacías
                        md_lines.append("| " + " | ".join(values) + " |")
                
                sheet_text = f"Contenido de la hoja '{sheetname}':\n" + "\n".join(md_lines)
                add_chunk(sheet_text, meta, texts, metas)
            except Exception as e:
                logging.warning(f"No se pudo procesar la hoja '{sheetname}' en '{basename}' como Markdown, usando método de respaldo: {e}")
                fallback_text = "\n".join([",".join([str(cell.value or "") for cell in row]) for row in rows])
                if fallback_text.strip():
                    add_chunk(fallback_text, meta, texts, metas)

    except Exception as e:
        logging.error(f"Error procesando XLSX '{basename}': {e}")
    return texts, metas

def process_image(path: str) -> tuple[list[str], list[dict]]:
    texts, metas = [], []
    basename = os.path.basename(path)
    year = extract_year(basename)
    try:
        with open(path, "rb") as f:
            img_bytes = f.read()
        vision_text = vision_extract(img_bytes)
        if vision_text:
            add_chunk(vision_text, {"source": basename, "doc_year": year}, texts, metas)
    except Exception as e:
        logging.warning(f"Error procesando imagen '{basename}': {e}")
    return texts, metas

def process_file(path: str) -> tuple[list[str], list[dict]]:
    cache_f = os.path.join(CACHE_DIR, os.path.basename(path) + ".json")
    if os.path.exists(cache_f) and os.path.getmtime(cache_f) > os.path.getmtime(path):
        with open(cache_f, 'r', encoding="utf-8") as f:
            cache = json.load(f)
        return cache["texts"], cache["metadatas"]
    
    logging.info(f"Procesando fichero: {os.path.basename(path)}")
    ext = path.lower().split('.')[-1]
    processor_map = {
        "pdf": process_pdf, "pptx": process_pptx, "docx": process_docx,
        "xlsx": process_xlsx, "png": process_image, "jpg": process_image, "jpeg": process_image
    }
    
    texts_res, metas_res = [], []
    if ext in processor_map:
        texts_res, metas_res = processor_map[ext](path)
    
    if texts_res:
        os.makedirs(os.path.dirname(cache_f), exist_ok=True)
        with open(cache_f, "w", encoding="utf-8") as f:
            json.dump({"texts": texts_res, "metadatas": metas_res}, f, ensure_ascii=False, indent=2)
    return texts_res, metas_res

if __name__ == "__main__":
    print("── Construyendo índice vectorial v18 (Comprensivo) ──")
    os.makedirs(CACHE_DIR, exist_ok=True)
    extensions = ("*.pdf", "*.pptx", "*.docx", "*.xlsx", "*.png", "*.jpg", "*.jpeg")
    files = [f for ext in extensions for f in glob.glob(os.path.join(DOCS_FOLDER, f"**/{ext}"), recursive=True)]
    
    print(f"Se encontraron {len(files)} documentos. Procesando con {MAX_WORKERS} workers…")
    all_texts, all_metas = [], []
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futs = {pool.submit(process_file, fp): fp for fp in files}
        for fut in tqdm(as_completed(futs), total=len(futs), desc="Procesando Documentos"):
            try:
                texts, metas = fut.result()
                all_texts.extend(texts)
                all_metas.extend(metas)
            except Exception as e:
                logging.error(f"❌ Error crítico procesando {futs[fut]}: {e}", exc_info=True)
    
    # Eliminar duplicados exactos antes de indexar
    unique_texts: Dict[int, tuple[str, dict]] = {}
    for text, meta in zip(all_texts, all_metas):
        h = hash(text)
        if h not in unique_texts:
            unique_texts[h] = (text, meta)
    
    final_texts = [item[0] for item in unique_texts.values()]
    final_metas = [item[1] for item in unique_texts.values()]
    
    print(f"Se generaron {len(all_texts)} chunks en total ({len(final_texts)} únicos).")
    
    if os.path.exists(PERSIST_DIR):
        print(f"Borrando el directorio de índice antiguo: {PERSIST_DIR}")
        shutil.rmtree(PERSIST_DIR)
        
    if final_texts:
        print(f"Creando nuevo índice ChromaDB con {len(final_texts)} chunks...")
        Chroma.from_texts(
            texts=final_texts,
            embedding=embedder,
            metadatas=final_metas,
            persist_directory=PERSIST_DIR
        )
        print(f"✅ Índice vectorial creado con éxito en '{PERSIST_DIR}'.")
    else:
        print("❌ Advertencia: No se ha indexado ningún contenido. Revisa los logs y la carpeta 'docs'.")