# app.py (v15 - Compatible con Extracción Semántica)
# Objetivo: Adaptar el prompt para que aproveche al máximo la base de conocimiento
# generada en formato de Preguntas y Respuestas (Q&A) por build_index.py.

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

# 2. Prompt Principal (v15 - Adaptado para Q&A)
RAG_PROMPT_TEMPLATE = """
**TU MISIÓN:** Eres HRCiklum, un asistente de IA experto y muy preciso. Tu objetivo es responder a las preguntas de los empleados utilizando una base de conocimiento pre-procesada en formato de Preguntas y Respuestas (Q&A).

**REGLAS DE ORO (INVIOLABLES):**
1.  **IDIOMA:** Responde **siempre** en el mismo idioma de la PREGUNTA DEL USUARIO.
2.  **TONO:** Sé amigable, profesional y servicial.
3.  **FUNCIONAMIENTO INTERNO (MUY IMPORTANTE):** El CONTEXTO que se te proporciona NO es el texto original de los documentos. Es una lista de pares de Preguntas (P:) y Respuestas (R:) que ya ha sido generada por una IA. Tu única tarea es encontrar la pregunta (P:) en el CONTEXTO que mejor coincida con la PREGUNTA DEL USUARIO y devolver la respuesta (R:) correspondiente de la forma más natural posible.
4.  **SÍNTESIS (si es necesario):** Si varias preguntas y respuestas del contexto son relevantes para la pregunta del usuario, puedes combinar el contenido de varias respuestas (R:) para crear una contestación más completa, pero siempre basándote en la información que contienen.
5.  **SI NO ENCUENTRAS COINCIDENCIA, ESCALA:** Si ninguna de las preguntas (P:) en el CONTEXTO se parece a la pregunta del usuario, responde: "He revisado la documentación disponible, pero no he encontrado una respuesta directa a tu consulta. Para darte la información más precisa, te recomiendo que lo consultes directamente con el departamento de RRHH."
6.  **NO INVENTES:** No añadas información que no esté en las respuestas (R:) del contexto. Limítate a presentar la información encontrada de una manera conversacional.

**CONTEXTO (Formato Q&A):**
{context}

---
**PREGUNTA DEL USUARIO (ya contextualizada con el historial):**
{input}

**TU RESPUESTA (basada en las 'R:' del contexto):**
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

    # El retriever ahora buscará en un espacio de Q&A, por lo que los resultados serán más precisos.
    base_retriever = vector_store.as_retriever(search_kwargs={"k": 10}) # k=10 es un buen punto de partida para Q&A

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
    logging.info("✅ Arquitectura de IA Conversacional (v15 - Semántica) inicializada correctamente.")

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
