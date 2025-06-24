# build_index.py (Versión Final)
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
import docx
import openpyxl
from concurrent.futures import ProcessPoolExecutor, as_completed
from io import BytesIO
from typing import List
from tqdm import tqdm
from PIL import Image
from pdf2image import convert_from_path
from dotenv import load_dotenv
from transformers import AutoTokenizer
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.schema.messages import HumanMessage, SystemMessage
from langchain_chroma import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter

# --- Configuración Global ---
load_dotenv()
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
vision_llm = ChatOpenAI(model=VISION_MODEL, openai_api_base=API_BASE, openai_api_key=API_KEY, max_tokens=4096)
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
    except Exception:
        return ""

def add_chunk(text_content: str, meta: dict, texts: list[str], metas: list[dict]):
    cleaned_content = BOILERPLATE_PAT.sub("", text_content).strip()
    if not cleaned_content:
        return
    chunks = text_splitter.split_text(cleaned_content)
    for chunk in chunks:
        enriched_meta = meta.copy()
        enriched_chunk = f"PREGUNTAS HIPOTÉTICAS QUE ESTE TEXTO PODRÍA RESPONDER:\n{generate_questions(chunk)}\n---\nCONTENIDO DEL DOCUMENTO:\n{chunk}"
        texts.append(enriched_chunk)
        metas.append(enriched_meta)

def extract_text_from_pdf(path: str) -> list[tuple[str, int]]:
    results = []
    with fitz.open(path) as doc:
        for i, page in enumerate(doc, 1):
            text = page.get_text("text").strip()
            results.append((text, i))
    return results

def vision_extract(img_bytes: bytes) -> str:
    def handler(signum, frame): raise TimeoutError
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

# (El resto de funciones de procesamiento de ficheros se mantienen igual)
def process_pdf(path: str): # ... (sin cambios)
    texts, metas = [], []
    basename = os.path.basename(path)
    year = extract_year(basename)
    raw_pages = extract_text_from_pdf(path)
    for page_text, page_num in raw_pages:
        meta = {"source": basename, "page": page_num, "doc_year": year}
        if page_text:
            add_chunk(page_text, meta, texts, metas)
        else:
            try:
                images = convert_from_path(path, first_page=page_num, last_page=page_num)
                if images:
                    buf = BytesIO()
                    images[0].save(buf, format="PNG")
                    vision_text = vision_extract(buf.getvalue())
                    add_chunk(vision_text, meta, texts, metas)
            except Exception as e:
                logging.warning(f"Error en OCR para pág {page_num} de {basename}: {e}")
    return texts, metas

def process_pptx(path: str): # ... (sin cambios)
    texts, metas = [], []
    basename = os.path.basename(path)
    year = extract_year(basename)
    prs = pptx.Presentation(path)
    for i, slide in enumerate(prs.slides, 1):
        slide_text = "\n".join([sh.text for sh in slide.shapes if hasattr(sh, "text")]).strip()
        if slide_text:
            add_chunk(slide_text, {"source": basename, "slide": i, "doc_year": year}, texts, metas)
    return texts, metas

def process_docx(path: str): # ... (sin cambios)
    texts, metas = [], []
    basename = os.path.basename(path)
    year = extract_year(basename)
    doc = docx.Document(path)
    full_text = "\n\n".join([p.text for p in doc.paragraphs if p.text.strip()])
    if full_text:
        add_chunk(full_text, {"source": basename, "doc_year": year}, texts, metas)
    return texts, metas

def process_xlsx(path: str): # ... (sin cambios)
    texts, metas = [], []
    basename = os.path.basename(path)
    year = extract_year(basename)
    wb = openpyxl.load_workbook(path, data_only=True)
    for sheetname in wb.sheetnames:
        sheet = wb[sheetname]
        sheet_text = "\n".join([",".join([str(cell.value or "") for cell in row]) for row in sheet.iter_rows()])
        if sheet_text.strip():
            add_chunk(sheet_text, {"source": basename, "sheet": sheetname, "doc_year": year}, texts, metas)
    return texts, metas

def process_image(path: str): # ... (sin cambios)
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
        logging.warning(f"Error procesando imagen {basename}: {e}")
    return texts, metas

def process_file(path: str) -> tuple[list[str], list[dict]]:
    cache_f = os.path.join(CACHE_DIR, os.path.basename(path) + ".json")
    if os.path.exists(cache_f) and os.path.getmtime(cache_f) > os.path.getmtime(path):
        with open(cache_f, encoding="utf-8") as f:
            cache = json.load(f)
        return cache["texts"], cache["metadatas"]
    ext = path.lower().split('.')[-1]
    processor_map = {
        "pdf": process_pdf, "pptx": process_pptx, "docx": process_docx,
        "xlsx": process_xlsx, "png": process_image, "jpg": process_image, "jpeg": process_image
    }
    if ext in processor_map:
        t, m = processor_map[ext](path)
    else:
        t, m = [], []
    if t:
        os.makedirs(os.path.dirname(cache_f), exist_ok=True)
        with open(cache_f, "w", encoding="utf-8") as f:
            json.dump({"texts": t, "metadatas": m}, f, ensure_ascii=False)
    return t, m

if __name__ == "__main__":
    print("── Construyendo índice vectorial v17 (Robusto) ──")
    os.makedirs(CACHE_DIR, exist_ok=True)
    extensions = ("*.pdf", "*.pptx", "*.docx", "*.xlsx", "*.png", "*.jpg", "*.jpeg")
    files = [f for ext in extensions for f in glob.glob(os.path.join(DOCS_FOLDER, ext))]
    print(f"Se encontraron {len(files)} documentos. Procesando con {MAX_WORKERS} workers…")
    all_texts, all_metas = [], []
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futs = {pool.submit(process_file, fp): fp for fp in files}
        for fut in tqdm(as_completed(futs), total=len(futs), desc="Documentos"):
            try:
                texts, metas = fut.result()
                all_texts.extend(texts)
                all_metas.extend(metas)
            except Exception as e:
                print(f"❌ Error procesando {futs[fut]}: {e}")
    
    unique_texts, unique_metas = [], []
    seen_hashes = set()
    for text, meta in zip(all_texts, all_metas):
        h = hash(text)
        if h not in seen_hashes:
            unique_texts.append(text)
            unique_metas.append(meta)
            seen_hashes.add(h)
    
    print(f"Se generaron {len(all_texts)} chunks en total. Quedan {len(unique_texts)} únicos.")
    if os.path.exists(PERSIST_DIR):
        print(f"Borrando el directorio de índice antiguo: {PERSIST_DIR}")
        shutil.rmtree(PERSIST_DIR)
    if unique_texts:
        print(f"Creando nuevo índice con {len(unique_texts)} chunks...")
        Chroma.from_texts(
            texts=unique_texts,
            embedding=embedder,
            metadatas=unique_metas,
            persist_directory=PERSIST_DIR
        )
        print(f"✅ Índice vectorial creado con éxito en '{PERSIST_DIR}'.")
    else:
        print("❌ No se ha indexado ningún contenido.")
