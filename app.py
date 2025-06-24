# app.py (v14 - Asistente Experto con Cita de Fuentes)
# Objetivo: Aumentar drásticamente la fiabilidad y utilidad del bot mediante
# un retriever con compresión contextual y la cita explícita de fuentes.

import os
import logging
from flask import Flask, request, jsonify
from dotenv import load_dotenv
from typing import List, Dict, Any

# --- Importaciones Clave de LangChain y OpenAI ---
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.documents import Document
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_core.output_parsers import StrOutputParser
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import create_history_aware_retriever
# === NUEVAS IMPORTACIONES PARA EL RETRIEVER INTELIGENTE ===
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
# ==========================================================
from langchain_core.messages import HumanMessage, AIMessage

# --- Configuración Inicial del Servidor ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
load_dotenv()

# --- Constantes y Variables de Entorno ---
API_KEY = os.getenv("OPENAI_API_KEY")
BASE_URL = os.getenv("OPENAI_API_BASE", "https://genai-gateway.azure-api.net/")
PERSIST_DIRECTORY = "chroma_db"
MODEL_NAME = "gpt-4o"
EMBEDDING_MODEL_NAME = "text-embedding-3-large"

if not API_KEY:
    logging.critical("FATAL: No se ha encontrado la 'OPENAI_API_KEY'. La aplicación no puede iniciarse.")
else:
    logging.info("La configuración de la API se ha cargado correctamente.")

# --- "Memoria" del Chatbot ---
chat_histories = {}

# --- Plantillas de Prompt: El Corazón del Asistente ---

# 1. Prompt para Contextualizar la Pregunta (sin cambios)
CONTEXTUALIZE_PROMPT_TEMPLATE = """
Dada la siguiente conversación (chat_history) y la última pregunta del usuario (input), reformula la pregunta para que sea una pregunta independiente y clara que pueda entenderse sin el historial previo. No respondas a la pregunta, únicamente reformúlala.
"""

# 2. Prompt Principal (v14 - Tono Experto y Cita de Fuentes)
RAG_PROMPT_TEMPLATE_V2 = """
**TU MISIÓN:** Eres HRCiklum, un asistente de IA experto en RRHH para los empleados de Ciklum. Tu personalidad es la de un colega senior: extremadamente competente, fiable, servicial y amigable. Tu objetivo es proporcionar respuestas precisas y útiles, basándote **estrictamente** en el CONTEXTO proporcionado.

**PLAN DE ACCIÓN (OBLIGATORIO):**
1.  **ANÁLISIS DE LA PREGUNTA:** Comprende profundamente la necesidad del usuario. ¿Qué información específica está buscando?
2.  **SÍNTESIS DE LA INFORMACIÓN:** Revisa el CONTEXTO. No te limites a copiar y pegar. Sintetiza la información de los diferentes fragmentos para construir una respuesta coherente y completa, como lo haría un experto.
3.  **RESPUESTA DIRECTA Y CLARA:** Responde a la pregunta del usuario en su mismo idioma. Ve al grano, pero sé amable.
4.  **CITA TUS FUENTES:** Al final de tu respuesta, DEBES indicar de dónde has sacado la información. Utiliza los metadatos del contexto (source, page, slide) para construir una referencia clara. Por ejemplo: *"(Fuente: Guia_Beneficios_2024.pdf, pág. 12)"*. Si usas varias fuentes, cítalas todas.
5.  **GESTIÓN DE LA INCERTIDUMBRE:**
    * Si el CONTEXTO contiene información relacionada pero no responde directamente a la pregunta, explica lo que sí has encontrado y guía al usuario. Ejemplo: "No he encontrado los detalles exactos sobre el seguro para padres, pero la póliza general indica que se puede extender la cobertura a familiares. Para confirmar las condiciones y el coste, te recomiendo contactar directamente con RRHH."
    * **NUNCA** digas "No he encontrado la información" si el contexto te da pistas, por mínimas que sean. Tu deber es ser útil.
    * Solo si el CONTEXTO está completamente vacío o no tiene relación alguna, usa la frase: "He revisado la documentación disponible, pero no he encontrado una respuesta a tu consulta. Te recomiendo que lo consultes directamente con el departamento de RRHH para darte la información más precisa."

**CONTEXTO (Información extraída de los documentos internos de Ciklum):**
{context}

---
**PREGUNTA DEL USUARIO (ya contextualizada con el historial):**
{input}

**TU RESPUESTA (siguiendo tu Plan de Acción y citando las fuentes):**
"""

# --- Arquitectura de la Cadena de IA (Experta y Fiable) ---
chain = None
try:
    llm = ChatOpenAI(model_name=MODEL_NAME, temperature=0.0, openai_api_base=BASE_URL, openai_api_key=API_KEY)
    embedder = OpenAIEmbeddings(model=EMBEDDING_MODEL_NAME, openai_api_base=BASE_URL, openai_api_key=API_KEY)

    vector_store = Chroma(persist_directory=PERSIST_DIRECTORY, embedding_function=embedder)

    item_count = vector_store._collection.count()
    logging.info(f"✅ La base de datos se ha cargado con {item_count} documentos.")
    if item_count == 0:
        logging.warning("ADVERTENCIA: La base de datos se ha cargado pero está vacía. Ejecuta build_index.py.")

    # *** CAMBIO 1: RETRIEVER CON FILTRADO INTELIGENTE (COMPRESSOR) ***
    # El retriever base busca los documentos iniciales (k=10).
    base_retriever = vector_store.as_retriever(search_kwargs={"k": 10, "score_threshold": 0.5})
    # El compresor usa un LLM para "releer" los documentos y extraer solo lo relevante.
    document_compressor = LLMChainExtractor.from_llm(llm)
    # Combinamos ambos para crear un recuperador mucho más preciso.
    contextual_compression_retriever = ContextualCompressionRetriever(
        base_compressor=document_compressor,
        base_retriever=base_retriever
    )

    def format_docs_with_sources(docs: List[Document]) -> str:
        """Formatea los documentos para el prompt, manteniendo los metadatos visibles."""
        formatted_docs = []
        for i, doc in enumerate(docs):
            # Creamos una cabecera para cada fragmento de contexto con sus metadatos
            meta = doc.metadata or {}
            source = meta.get('source', 'Desconocido')
            page = meta.get('page')
            slide = meta.get('slide')
            source_info = f"Fuente: {source}"
            if page:
                source_info += f", pág. {page}"
            if slide:
                source_info += f", slide {slide}"

            # Unimos la metadata con el contenido
            formatted_docs.append(f"--- Fragmento de Contexto {i+1} ---\nMetadata: {source_info}\nContenido: {doc.page_content}\n--- Fin del Fragmento ---")

        return "\n\n".join(formatted_docs)

    def get_sources_from_docs(docs: List[Document]) -> str:
        """Extrae y formatea las fuentes para la respuesta final."""
        if not docs:
            return ""
        
        sources = []
        for doc in docs:
            meta = doc.metadata or {}
            source = meta.get('source', 'N/A')
            page = meta.get('page')
            slide = meta.get('slide')
            
            ref = f"{source}"
            if page: ref += f", pág. {page}"
            if slide: ref += f", slide {slide}"
            if ref not in sources:
                sources.append(ref)
        
        return "\n\n*Fuente(s): " + ", ".join(sources) + "*" if sources else ""


    contextualize_q_prompt = ChatPromptTemplate.from_messages([
        ("system", CONTEXTUALIZE_PROMPT_TEMPLATE),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
    ])
    
    history_aware_retriever = create_history_aware_retriever(llm, contextual_compression_retriever, contextualize_q_prompt)

    # *** CAMBIO 2: PROMPT DE EXPERTO (V2) ***
    answer_generation_prompt = ChatPromptTemplate.from_messages([
        ("system", RAG_PROMPT_TEMPLATE_V2),
        ("human", "{input}"),
    ])

    # *** CAMBIO 3: CADENA CON PROCESAMIENTO PARALELO PARA OBTENER RESPUESTA Y FUENTES ***
    rag_chain = (
        RunnablePassthrough.assign(
            context_docs=history_aware_retriever
        ).assign(
            context=lambda x: format_docs_with_sources(x["context_docs"]),
            sources=lambda x: get_sources_from_docs(x["context_docs"])
        )
        .assign(
            answer=(
                answer_generation_prompt
                | llm
                | StrOutputParser()
            )
        )
    )

    chain = rag_chain
    logging.info("✅ Arquitectura de IA Experta (v14) inicializada correctamente.")

except Exception as e:
    logging.critical(f"❌ FATAL: La cadena RAG no pudo inicializarse: {e}", exc_info=True)
    chain = None

# --- Aplicación Web Flask ---
app = Flask(__name__)

@app.route("/chat", methods=["POST"])
def handle_chat_event():
    if not chain:
        return jsonify({"text": "Lo siento, el asistente no está disponible en este momento. Por favor, revisa los logs del servidor."}), 500

    data = request.json
    user_input = data.get('message', {}).get('text', '').strip()
    session_id = data.get('user', {}).get('id', 'default_session')

    if not user_input or data.get('user', {}).get('type') == 'BOT':
        return jsonify({})

    logging.info(f"Consulta recibida de '{session_id}': '{user_input}'")
    current_chat_history = chat_histories.get(session_id, [])

    try:
        # La cadena ahora devuelve un diccionario con 'answer', 'sources', etc.
        result = chain.invoke({"input": user_input, "chat_history": current_chat_history})
        
        answer = result["answer"]
        sources = result["sources"]

        # Combinamos la respuesta y las fuentes para la salida final
        final_response = f"{answer}{sources}"

        current_chat_history.extend([
            HumanMessage(content=user_input),
            AIMessage(content=final_response) # Guardamos la respuesta completa en el historial
        ])
        chat_histories[session_id] = current_chat_history[-10:]

        logging.info(f"Respuesta generada para '{session_id}': '{final_response}'")
        return jsonify({"text": final_response})

    except Exception as e:
        logging.error(f"Error procesando la solicitud RAG: {e}", exc_info=True)
        return jsonify({"text": "Lo siento, ha ocurrido un error al procesar tu solicitud."}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(port=port, host='0.0.0.0', debug=True)