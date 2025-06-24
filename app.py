# app.py (v13 - Solución Fiable y Consistente)
# Objetivo: Utilizar un retriever simplificado y un prompt con "Obligación de Responder"
# para eliminar la inconsistencia y asegurar que el bot sea siempre útil.

import os
import logging
from flask import Flask, request, jsonify
from dotenv import load_dotenv
from typing import List, Dict, Any

# --- Importaciones Clave de LangChain y OpenAI ---
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.documents import Document
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import create_history_aware_retriever
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

# 1. Prompt para Contextualizar la Pregunta con el Historial.
CONTEXTUALIZE_PROMPT_TEMPLATE = """
Dada la siguiente conversación (chat_history) y la última pregunta del usuario (input), reformula la pregunta para que sea una pregunta independiente y clara que pueda entenderse sin el historial previo. No respondas a la pregunta, únicamente reformúlala.
"""

# 2. Prompt Principal (v13 - Con Plan de Razonamiento Robusto)
RAG_PROMPT_TEMPLATE = """
**TU MISIÓN:** Eres HRCiklum, un asistente de IA experto, fiable y muy servicial para los empleados de Ciklum. Tu objetivo es responder siempre de forma útil a las preguntas, basándote **única y exclusivamente** en la información del CONTEXTO proporcionado.

**REGLAS DE ORO (INVIOLABLES):**
1.  **IDIOMA:** Responde **siempre** en el mismo idioma de la PREGUNTA DEL USUARIO.
2.  **TONO:** Sé amigable, profesional y servicial.
3.  **BASE EN EL CONTEXTO:** Tu conocimiento se limita **estrictamente** al CONTEXTO. No inventes información.
4.  **SI NO SABES, ESCALA:** Solo si después de seguir tu plan de razonamiento no encuentras una respuesta clara en el CONTEXTO, responde: "He revisado la documentación disponible, pero no he encontrado una respuesta directa a tu consulta. Para darte la información más precisa, te recomiendo que lo consultes directamente con el departamento de RRHH."

**PLAN DE RAZONAMIENTO (OBLIGATORIO):**
Antes de escribir tu respuesta, sigue mentalmente estos 5 pasos para asegurar la máxima calidad:
1.  **IDENTIFICAR EL TEMA CENTRAL:** ¿Cuál es el tema principal y específico sobre el que pregunta el usuario? (p.ej., 'proceso para el examen de salud', 'proveedor del seguro médico').
2.  **BÚSQUEDA ENFOCADA:** Del CONTEXTO proporcionado, localiza todos los fragmentos que traten directamente sobre ese TEMA CENTRAL.
3.  **FILTRADO INTELIGENTE:** Ignora la información de los fragmentos que, aunque estén relacionados, no respondan directamente a la pregunta del usuario. (Ejemplo: si la pregunta es sobre el 'seguro médico', ignora las menciones a 'formación de riesgos laborales').
4.  **CONSTRUCCIÓN DE LA RESPUESTA:** Usa la información que has filtrado para redactar una respuesta clara, concisa y útil.
5.  **OBLIGACIÓN DE RESPONDER (REGLA CRÍTICA):** Si en el paso 3 has encontrado información relevante que responde, aunque sea parcialmente, a la pregunta del usuario, **DEBES** proporcionar esa respuesta. **NO utilices** la respuesta de "No he encontrado la información" si el contexto contiene datos útiles. Tu objetivo es ser siempre de ayuda.

**CONTEXTO DE LOS DOCUMENTOS:**
{context}

---
**PREGUNTA DEL USUARIO (ya contextualizada con el historial):**
{input}

**TU RESPUESTA (siguiendo el Plan de Razonamiento):**
"""

# --- Arquitectura de la Cadena de IA (Simplificada y Robusta) ---
chain = None
try:
    llm = ChatOpenAI(model_name=MODEL_NAME, temperature=0.0, openai_api_base=BASE_URL, openai_api_key=API_KEY)
    embedder = OpenAIEmbeddings(model=EMBEDDING_MODEL_NAME, openai_api_base=BASE_URL, openai_api_key=API_KEY)

    vector_store = Chroma(persist_directory=PERSIST_DIRECTORY, embedding_function=embedder)

    item_count = vector_store._collection.count()
    logging.info(f"✅ La base de datos se ha cargado con {item_count} documentos.")
    if item_count == 0:
        logging.warning("ADVERTENCIA: La base de datos se ha cargado pero está vacía.")

    # *** CAMBIO 1: RETRIEVER SIMPLIFICADO Y FIABLE ***
    # Usamos el retriever base directamente para eliminar la inconsistencia del MultiQueryRetriever.
    # Mantenemos k=15 para asegurar que tenga suficiente contexto para responder.
    base_retriever = vector_store.as_retriever(search_kwargs={"k": 15})

    def format_docs(docs: List[Document]) -> str:
        return "\n\n".join(doc.page_content for doc in docs)

    contextualize_q_prompt = ChatPromptTemplate.from_messages([
        ("system", CONTEXTUALIZE_PROMPT_TEMPLATE),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
    ])
    
    # Usamos el 'base_retriever' en lugar del 'MultiQueryRetriever'
    history_aware_retriever = create_history_aware_retriever(llm, base_retriever, contextualize_q_prompt)

    # *** CAMBIO 2: PROMPT CON OBLIGACIÓN DE RESPONDER ***
    answer_generation_prompt = ChatPromptTemplate.from_messages([
        ("system", RAG_PROMPT_TEMPLATE),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
    ])

    rag_chain = (
        RunnablePassthrough.assign(
            context=history_aware_retriever
        ).assign(
            answer=(
                RunnablePassthrough.assign(
                    context=lambda x: format_docs(x["context"])
                )
                | answer_generation_prompt
                | llm
                | StrOutputParser()
            )
        )
    )

    chain = rag_chain
    logging.info("✅ Arquitectura de IA Conversacional (v13) inicializada correctamente.")

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
        result = chain.invoke({"input": user_input, "chat_history": current_chat_history})
        answer = result["answer"]

        current_chat_history.extend([
            HumanMessage(content=user_input),
            AIMessage(content=answer)
        ])
        chat_histories[session_id] = current_chat_history[-10:]

        return jsonify({"text": answer})

    except Exception as e:
        logging.error(f"Error procesando la solicitud RAG: {e}", exc_info=True)
        return jsonify({"text": "Lo siento, ha ocurrido un error al procesar tu solicitud."}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(port=port, host='0.0.0.0', debug=True)
