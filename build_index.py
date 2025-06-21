# build_index.py  ─── Versión 2025‑06‑21 (Multiformato)
"""
Reconstruye el índice vectorial de la base documental de Ciklum.
Ahora soporta: PDF, PowerPoint, Word, Excel e Imágenes (PNG, JPG).
"""

from __future__ import annotations

import os
import re
import glob
import json
import shutil
import base64
import signal
import fitz  # PyMuPDF
import pptx
import docx # Para Word
import openpyxl # Para Excel
from concurrent.futures import ProcessPoolExecutor, as_completed
from io import BytesIO
from typing import List, Tuple

from tqdm import tqdm
from PIL import Image # Para imágenes
from pdf2image import convert_from_path
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.schema.messages import HumanMessage, SystemMessage
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter

# ── Configuración global ────────────────────────────────────────────────────
load_dotenv()
VISION_MODEL = "gpt-4o"
TEXT_MODEL = "gpt-4o"
DOCS_FOLDER = "docs"
CACHE_DIR = ".cache/doc_cache"
PERSIST_DIR = "chroma_db"
CHUNK_SIZE = 4096
CHUNK_OVERLAP = 200
MAX_WORKERS = min(os.cpu_count() or 4, 4)
VISION_TIMEOUT = 90  # segundos por página

API_BASE = "https://genai-gateway.azure-api.net/"
API_KEY = os.getenv("OPENAI_API_KEY")

# ── Modelos ────────────────────────────────────────────────────────────────
vision_llm = ChatOpenAI(model=VISION_MODEL, openai_api_base=API_BASE, openai_api_key=API_KEY, max_tokens=4096)
text_llm = ChatOpenAI(model=TEXT_MODEL, openai_api_base=API_BASE, openai_api_key=API_KEY, max_tokens=2048)
embedder = OpenAIEmbeddings(model="text-embedding-3-large", openai_api_base=API_BASE, openai_api_key=API_KEY)
text_splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)

# ── Utilidades ─────────────────────────────────────────────────────────────
YEAR_PAT = re.compile(r"(20\d{2})")
ROW_BAR = re.compile(r"\|.*\|")


def extract_year(filename: str) -> int | None:
    m = YEAR_PAT.search(filename)
    return int(m.group(1)) if m else None


def generate_questions(chunk: str) -> str:
    try:
        resp = text_llm.invoke([
            SystemMessage(content="Genera 3 preguntas complejas que un empleado podría hacer cuya respuesta esté en el fragmento"),
            HumanMessage(content=chunk)
        ])
        return resp.content.strip()
    except Exception:
        return ""


def add_chunk(text: str, meta: dict, texts: list[str], metas: list[dict]):
    enriched = f"PREGUNTAS:\n{generate_questions(text)}\n---\n{text}"
    texts.append(enriched)
    metas.append(meta)


def split_table_rows(page_text: str) -> list[tuple[str, dict]]:
    rows: list[tuple[str, dict]] = []
    buffer: list[str] = []
    header: list[str] = []
    for ln in page_text.splitlines():
        if ROW_BAR.match(ln):
            if not header:
                header.append(ln)
                continue
            buffer.append(ln)
            cols = [c.strip() for c in ln.split("|") if c.strip()]
            if cols:
                rows.append(("\n".join(header + [ln]), {"row": cols[0]}))
        else:
            header, buffer = [], []
    return rows


# ── Extracción de textos ───────────────────────────────────────────────────

def extract_text_from_pdf(path: str) -> list[tuple[str, int]]:
    """Devuelve lista de (texto, nº de página). Usa PyMuPDF; si la página está vacía ⇒ None."""
    results = []
    with fitz.open(path) as doc:
        for i, page in enumerate(doc, 1):
            text = page.get_text("text").strip()
            results.append((text, i))
    return results


def vision_extract(img_bytes: bytes) -> str:
    def handler(signum, frame):
        raise TimeoutError
    signal.signal(signal.SIGALRM, handler)
    signal.alarm(VISION_TIMEOUT)
    try:
        b64 = base64.b64encode(img_bytes).decode()
        resp = vision_llm.invoke([
            SystemMessage(content="Extrae todo el texto y tablas como Markdown"),
            HumanMessage(content=[
                {"type": "text", "text": "Procesa la imagen"},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}}
            ])
        ])
        return resp.content
    finally:
        signal.alarm(0)


def process_pdf(path: str) -> tuple[list[str], list[dict]]:
    texts, metas = [], []
    basename = os.path.basename(path)
    year = extract_year(basename)

    raw_pages = extract_text_from_pdf(path)
    # convert pages lacking text
    for page_text, page_num in tqdm(raw_pages, desc=f"{basename}"):
        if page_text:
            add_chunk(page_text, {"source": basename, "page": page_num, "doc_year": year}, texts, metas)
            for row_text, row_meta in split_table_rows(page_text):
                add_chunk(row_text, {"source": basename, "page": page_num, "doc_year": year, **row_meta}, texts, metas)
            continue
        # page likely scanned → use vision
        try:
            png = convert_from_path(path, first_page=page_num, last_page=page_num)[0]
            buf = BytesIO(); png.save(buf, format="PNG")
            vision_text = vision_extract(buf.getvalue())
            if vision_text:
                add_chunk(vision_text, {"source": basename, "page": page_num, "doc_year": year}, texts, metas)
                for row_text, row_meta in split_table_rows(vision_text):
                    add_chunk(row_text, {"source": basename, "page": page_num, "doc_year": year, **row_meta}, texts, metas)
        except TimeoutError:
            print(f"⚠️  Timeout visión pág {page_num} de {basename}")
        except Exception as e:
            print(f"⚠️  Error visión pág {page_num} de {basename}: {e}")
    return texts, metas


def process_pptx(path: str) -> tuple[list[str], list[dict]]:
    texts, metas = [], []
    basename = os.path.basename(path)
    year = extract_year(basename)
    prs = pptx.Presentation(path)
    for i, slide in enumerate(tqdm(prs.slides, desc=basename), 1):
        slide_text = "\n".join(
            [sh.text for sh in slide.shapes if getattr(sh, "text", "")]  # type: ignore[attr-defined]
        ).strip()
        if not slide_text:
            continue
        add_chunk(slide_text, {"source": basename, "slide": i, "doc_year": year}, texts, metas)
        for row_text, row_meta in split_table_rows(slide_text):
            add_chunk(row_text, {"source": basename, "slide": i, "doc_year": year, **row_meta}, texts, metas)
    return texts, metas

# --- NUEVAS FUNCIONES ---

def process_docx(path: str) -> tuple[list[str], list[dict]]:
    """Extrae texto de un documento de Word (.docx)."""
    texts, metas = [], []
    basename = os.path.basename(path)
    year = extract_year(basename)
    doc = docx.Document(path)
    full_text = "\n".join([para.text for para in doc.paragraphs])
    if full_text:
        add_chunk(full_text, {"source": basename, "doc_year": year}, texts, metas)
    return texts, metas

def process_xlsx(path: str) -> tuple[list[str], list[dict]]:
    """Extrae texto de cada hoja de un fichero Excel (.xlsx)."""
    texts, metas = [], []
    basename = os.path.basename(path)
    year = extract_year(basename)
    workbook = openpyxl.load_workbook(path)
    for sheetname in workbook.sheetnames:
        sheet = workbook[sheetname]
        sheet_text = "\n".join(
            [",".join([str(cell.value) for cell in row if cell.value is not None]) for row in sheet.iter_rows()]
        )
        if sheet_text:
            meta = {"source": basename, "sheet": sheetname, "doc_year": year}
            add_chunk(sheet_text, meta, texts, metas)
    return texts, metas

def process_image(path: str) -> tuple[list[str], list[dict]]:
    """Extrae texto de un fichero de imagen usando OCR (GPT-4o Vision)."""
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
        print(f"⚠️  Error procesando imagen {basename}: {e}")
    return texts, metas

# -------------------------

def process_file(path: str) -> tuple[list[str], list[dict]]:
    cache_f = os.path.join(CACHE_DIR, os.path.basename(path) + ".json")
    if os.path.exists(cache_f) and os.path.getmtime(cache_f) > os.path.getmtime(path):
        with open(cache_f, encoding="utf-8") as f:
            cache = json.load(f)
        return cache["texts"], cache["metadatas"]

    # --- LÓGICA ACTUALIZADA ---
    ext = path.lower().split('.')[-1]
    if ext == "pdf":
        t, m = process_pdf(path)
    elif ext == "pptx":
        t, m = process_pptx(path)
    elif ext == "docx":
        t, m = process_docx(path)
    elif ext == "xlsx":
        t, m = process_xlsx(path)
    elif ext in ["png", "jpg", "jpeg"]:
        t, m = process_image(path)
    else:
        t, m = [], []
    # ---------------------------

    if t:
        with open(cache_f, "w", encoding="utf-8") as f:
            json.dump({"texts": t, "metadatas": m}, f, ensure_ascii=False)
    return t, m


# ── Main ───────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("── Construyendo índice vectorial (multiformato) ──")
    os.makedirs(CACHE_DIR, exist_ok=True)

    # --- BÚSQUEDA DE FICHEROS ACTUALIZADA ---
    extensions_to_process = ("*.pdf", "*.pptx", "*.docx", "*.xlsx", "*.png", "*.jpg", "*.jpeg")
    files = [f for ext in extensions_to_process for f in glob.glob(os.path.join(DOCS_FOLDER, ext))]
    print(f"Se encontraron {len(files)} documentos. Procesando con {MAX_WORKERS} workers…")
    # ----------------------------------------

    all_texts, all_metas = [], []
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futs = {pool.submit(process_file, fp): fp for fp in files}
        for fut in tqdm(as_completed(futs), total=len(futs), desc="Documentos"):
            fp = futs[fut]
            try:
                t, m = fut.result()
                all_texts.extend(t); all_metas.extend(m)
            except Exception as e:
                print(f"❌ Error procesando {fp}: {e}")

    if os.path.exists(PERSIST_DIR):
        shutil.rmtree(PERSIST_DIR)

    if all_texts:
        Chroma.from_texts(all_texts, embedder, metadatas=all_metas, persist_directory=PERSIST_DIR)
        print(f"✅ Índice vectorial creado con {len(all_texts)} chunks.")
    else:
        print("❌ Sin contenido indexado. Comprueba los documentos.")