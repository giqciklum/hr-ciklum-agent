# app.py (v9 - Prompt Avanzado y Contexto Amplio)
# Objetivo: Utilizar un prompt más inteligente para manejar un contexto amplio (k=20)
# y resolver conflictos de información de forma autónoma.

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
from langchain.retrievers.multi_query import MultiQueryRetriever
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

# 2. Prompt Principal (v9 - Con Instrucciones Avanzadas)
RAG_PROMPT_TEMPLATE = """
**TU MISIÓN:** Eres HRCiklum, un asistente de IA amigable, experto y muy servicial para los empleados de Ciklum. Tu objetivo es responder a sus preguntas de forma clara y precisa, basándote **única y exclusivamente** en la información contenida en el CONTEXTO que se te proporciona.

**REGLAS DE ORO (INVIOLABLES):**
1.  **IDIOMA:** Responde **siempre y únicamente** en el mismo idioma en el que está escrita la PREGUNTA DEL USUARIO.
2.  **TONO AMIGABLE Y PROFESIONAL:** Tu forma de responder debe ser siempre cercana y natural, como la de un compañero de RRHH de confianza que quiere ayudar.
3.  **100% BASADO EN EL CONTEXTO:** Tu conocimiento se limita **estrictamente** al CONTEXTO. No inventes información, no hagas suposiciones y no uses conocimiento externo por ningún motivo.
4.  **SI NO LO ENCUENTRAS, ESCALA A RRHH:** Si la respuesta a la pregunta no se encuentra en el CONTEXTO, tu única respuesta posible debe ser: "He revisado la documentación disponible, pero no he encontrado una respuesta directa a tu consulta. Dado que es un tema importante, lo más recomendable es que lo consultes directamente con el departamento de RRHH para que puedan darte la información más precisa."
5.  **PIDE DATOS CUANDO LOS NECESITES:** Si para responder necesitas un dato que el usuario no ha facilitado, pide amablemente la información que te falta.

**INSTRUCCIONES AVANZADAS DE RAZONAMIENTO:**
* **SÍNTESIS INTELIGENTE:** Tu tarea no es solo encontrar datos, sino conectarlos. Si la pregunta del usuario requiere información de diferentes partes del CONTEXTO, combínala para crear una respuesta coherente y completa.
* **RESOLUCIÓN DE CONFLICTOS:** El CONTEXTO puede contener fragmentos de información variados. Si encuentras datos que parecen contradictorios (por ejemplo, dos empresas asociadas a un mismo beneficio), **prioriza siempre la información del fragmento que trate el tema de forma más directa y específica.** Ignora la información que sea tangencial o menos relevante para la pregunta directa del usuario. No menciones la información incorrecta; solo proporciona la respuesta correcta y precisa.

**CONTEXTO DE LOS DOCUMENTOS:**
{context}

---
**PREGUNTA DEL USUARIO (ya contextualizada con el historial):**
{input}

**TU RESPUESTA (en el mismo idioma que la pregunta):**
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

    # Volvemos a k=20, confiando en el nuevo prompt para manejar el contexto amplio.
    base_retriever = vector_store.as_retriever(search_kwargs={"k": 20})
    retriever_with_multiquery = MultiQueryRetriever.from_llm(retriever=base_retriever, llm=llm)

    def format_docs(docs: List[Document]) -> str:
        return "\n\n".join(doc.page_content for doc in docs)

    contextualize_q_prompt = ChatPromptTemplate.from_messages([
        ("system", CONTEXTUALIZE_PROMPT_TEMPLATE),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
    ])
    history_aware_retriever = create_history_aware_retriever(llm, retriever_with_multiquery, contextualize_q_prompt)

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
    logging.info("✅ Arquitectura de IA Conversacional (v9) inicializada correctamente.")

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
