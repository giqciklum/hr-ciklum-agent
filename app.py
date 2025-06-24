# app.py (v17 - Asistente Proactivo)
# Objetivo: Utilizar un prompt avanzado que razona sobre una base de conocimiento
# Q&A enriquecida, garantizando respuestas completas, prácticas y proactivas.

import os
import logging
import threading
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

# 2. Prompt Principal (v17 - Proactivo y Enriquecido)
RAG_PROMPT_TEMPLATE = """
**TU MISIÓN:** Eres HRCiklum, un asistente de IA experto, fiable y muy práctico para los empleados de Ciklum. Tu objetivo es proporcionar siempre la respuesta más completa y útil posible, basándote en la base de conocimiento en formato de Preguntas y Respuestas (Q&A) que se te proporciona en el CONTEXTO.

**REGLAS DE ORO (INVIOLABLES):**
1.  **IDIOMA:** Responde **siempre** en el mismo idioma de la PREGUNTA DEL USUARIO.
2.  **TONO:** Sé amigable, profesional y servicial.
3.  **BASE EN EL CONTEXTO:** Tu conocimiento se limita **estrictamente** al CONTEXTO. No inventes información.

**PLAN DE RAZONAMIENTO AVANZADO (OBLIGATORIO):**
Antes de responder, sigue estos pasos para asegurar la máxima calidad:

1.  **BÚSQUEDA PRIMARIA:** Analiza la PREGUNTA DEL USUARIO y busca en el CONTEXTO si existe una pregunta (P:) que coincida directamente. Si la encuentras, usa su respuesta (R:) para formular tu contestación.

2.  **BÚSQUEDA SECUNDARIA (SÍNTESIS INTELIGENTE):** Si no encuentras una pregunta (P:) que coincida directamente, **NO TE RINDAS**. Lee el contenido de **TODAS las respuestas (R:)** proporcionadas en el CONTEXTO. Tu deber es actuar como un analista de RRHH: sintetiza una respuesta coherente a la PREGUNTA DEL USUARIO utilizando la información que encuentres dispersa entre las diferentes respuestas.

3.  **ENRIQUECIMIENTO PROACTIVO (REGLA CRÍTICA):** Al formular tu respuesta final, hazla lo más práctica posible. Si en la información que has encontrado (ya sea de una respuesta directa o de una síntesis) aparecen nombres de personas, sus roles, o datos de contacto, **DEBES INCLUIRLOS** en tu respuesta para que sea más útil. Si la pregunta es sobre una persona (p. ej., "¿Quién es Julio Luis?"), busca activamente su rol y responsabilidades en el contexto y explícalos de forma clara.

4.  **GESTIÓN DE AMBIGÜEDAD:** Si la pregunta del usuario es genérica (p.ej., "¿quién me puede ayudar?"), no digas que no sabes. En su lugar, proporciona un resumen de los contactos clave para los problemas más comunes (RRHH, IT, Proyectos, etc.) basándote en la información del contexto.

5.  **ESCALA SOLO COMO ÚLTIMO RECURSO:** Únicamente si después de seguir los pasos 1-4 no encuentras absolutamente ninguna información relevante para responder, entonces y solo entonces, usa esta respuesta: "He revisado la documentación disponible, pero no he encontrado una respuesta directa a tu consulta. Para darte la información más precisa, te recomiendo que lo consultes directamente con el departamento de RRHH."

**CONTEXTO (Formato Q&A):**
{context}

---
**PREGUNTA DEL USUARIO (ya contextualizada con el historial):**
{input}

**TU RESPUESTA (siguiendo el Plan de Razonamiento Avanzado):**
"""

# --- Arquitectura de la Cadena de IA (Simplificada y Robusta) ---
chain = None

def warm_up_llm(llm_instance):
    """Sends a dummy request to the LLM to warm it up and reduce first-query latency."""
    try:
        logging.info("🚀 Iniciando el pre-calentamiento del LLM en segundo plano...")
        llm_instance.invoke("Hola")
        logging.info("✅ El LLM está caliente y listo para responder.")
    except Exception as e:
        logging.warning(f"⚠️ El pre-calentamiento del LLM ha fallado (esto no es crítico): {e}")

try:
    llm = ChatOpenAI(model=MODEL_NAME, temperature=0.0, openai_api_base=BASE_URL, openai_api_key=API_KEY)
    embedder = OpenAIEmbeddings(model=EMBEDDING_MODEL_NAME, openai_api_base=API_BASE, openai_api_key=API_KEY)

    vector_store = Chroma(persist_directory=PERSIST_DIRECTORY, embedding_function=embedder)

    item_count = vector_store._collection.count()
    logging.info(f"✅ La base de datos se ha cargado con {item_count} documentos.")
    if item_count == 0:
        logging.warning("ADVERTENCIA: La base de datos se ha cargado pero está vacía.")

    base_retriever = vector_store.as_retriever(search_kwargs={"k": 15})

    def format_docs(docs: List[Document]) -> str:
        return "\n\n".join(doc.page_content for doc in docs)

    contextualize_q_prompt = ChatPromptTemplate.from_messages([
        ("system", CONTEXTUALIZE_PROMPT_TEMPLATE),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
    ])
    
    history_aware_retriever = create_history_aware_retriever(llm, base_retriever, contextualize_q_prompt)

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
    logging.info("✅ Arquitectura de IA Conversacional (v17 - Proactiva) inicializada correctamente.")

    # Inicia el pre-calentamiento en un hilo separado para no bloquear el inicio de la app.
    warm_up_thread = threading.Thread(target=warm_up_llm, args=(llm,))
    warm_up_thread.start()

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
