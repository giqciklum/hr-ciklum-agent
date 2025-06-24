# app.py (v15 - Asistente Experto Fiable, sin Citas)
# Objetivo: Ser la fuente de verdad definitiva. Responde de forma natural y práctica,
# interpretando la intención del usuario para ser siempre útil. Corrige el bug de 'score_threshold'.

import os
import logging
from flask import Flask, request, jsonify
from dotenv import load_dotenv
from typing import List

# --- Importaciones Clave de LangChain y OpenAI ---
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.documents import Document
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import create_history_aware_retriever
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
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

# 2. Prompt Principal (v15 - El Experto Práctico y Fiable)
RAG_PROMPT_TEMPLATE_V3 = """
**TU ROL:** Eres HRCiklum, el asistente de RRHH de IA para los empleados de Ciklum. Tu identidad es la de un compañero experto: eres la persona a la que todos acuden porque das respuestas fiables, prácticas y fáciles de entender. Eres proactivo, servicial y tu credibilidad es tu mayor valor.

**TUS PRINCIPIOS (INQUEBRANTABLES):**
1.  **BASE EN LA EVIDENCIA:** Tus respuestas se basan **única y exclusivamente** en la información del CONTEXTO proporcionado. **NUNCA INVENTES NADA.** Tu reputación depende de tu veracidad.
2.  **INTERPRETA Y AYUDA:** No eres un simple buscador. Si un usuario pregunta algo general como "¿quién me puede ayudar?" o "tengo un problema", analiza la pregunta y el historial para inferir su necesidad real (ej: 'ayuda con la nómina', 'problema con las vacaciones'). Luego, busca en el CONTEXTO la persona, departamento o procedimiento correcto para resolver esa necesidad específica.
3.  **RESPUESTAS PRÁCTICAS Y DIRECTAS:** Comunícate de forma natural y clara, no como un robot. Estructura la información para que sea fácil de consumir: usa listas, puntos clave o pasos a seguir. El objetivo es que el empleado sepa exactamente qué hacer después de leer tu respuesta.
4.  **GESTIÓN DE LA INCERTIDUMBRE:**
    * Si el CONTEXTO tiene información parcial, úsala. Explica lo que sabes y guía al usuario sobre los siguientes pasos. Ejemplo: "Sobre el seguro para padres, la póliza general menciona la cobertura para familiares directos. Para conocer las condiciones exactas y el coste, el siguiente paso es contactar con el departamento de RRHH."
    * **NO** digas "No he encontrado la información" si tienes la más mínima pista. Tu deber es ser el compañero más útil de la empresa.
    * Solo si el CONTEXTO no contiene absolutamente nada relevante, responde: "He revisado toda la documentación interna y no he encontrado información sobre este tema. Para darte una respuesta precisa, lo mejor es que lo consultes directamente con el departamento de RRHH."

**CONTEXTO (Información interna y verificada de Ciklum):**
{context}

---
**PREGUNTA DEL USUARIO (previamente analizada y contextualizada):**
{input}

**TU RESPUESTA (clara, práctica y basada 100% en el contexto):**
"""

# --- Arquitectura de la Cadena de IA (Simplificada y Robusta) ---
chain = None
try:
    llm = ChatOpenAI(model_name=MODEL_NAME, temperature=0.0, openai_api_base=BASE_URL, openai_api_key=API_KEY)
    embedder = OpenAIEmbeddings(model=EMBEDDING_MODEL_NAME, openai_api_base=BASE_URL, openai_api_key=API_KEY)
    vector_store = Chroma(persist_directory=PERSIST_DIRECTORY, embedding_function=embedder)

    item_count = vector_store._collection.count()
    logging.info(f"✅ La base de datos se ha cargado con {item_count} chunks.")
    if item_count == 0:
        logging.warning("ADVERTENCIA: La base de datos está vacía. Ejecuta build_index.py con los chunks pequeños.")

    # *** CAMBIO 1: CORRECCIÓN DEL BUG - ELIMINADO 'score_threshold' ***
    # Este cambio soluciona el TypeError y evita que la app se caiga.
    base_retriever = vector_store.as_retriever(search_kwargs={"k": 15})
    
    document_compressor = LLMChainExtractor.from_llm(llm)
    contextual_compression_retriever = ContextualCompressionRetriever(
        base_compressor=document_compressor,
        base_retriever=base_retriever
    )

    # *** CAMBIO 2: SIMPLIFICACIÓN - FORMATEO SIMPLE SIN FUENTES ***
    def format_docs(docs: List[Document]) -> str:
        """Función simplificada para unir el contenido de los documentos."""
        return "\n\n".join(doc.page_content for doc in docs)

    contextualize_q_prompt = ChatPromptTemplate.from_messages([
        ("system", CONTEXTUALIZE_PROMPT_TEMPLATE),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
    ])
    
    history_aware_retriever = create_history_aware_retriever(llm, contextual_compression_retriever, contextualize_q_prompt)

    answer_generation_prompt = ChatPromptTemplate.from_messages([
        ("system", RAG_PROMPT_TEMPLATE_V3),
        ("human", "{input}"),
    ])

    # *** CAMBIO 3: CADENA MÁS LIMPIA Y DIRECTA ***
    # La cadena ahora se enfoca en una única tarea: generar la mejor respuesta posible.
    rag_chain = (
        {
            "context": history_aware_retriever | format_docs, 
            "input": RunnablePassthrough()
        }
        | answer_generation_prompt
        | llm
        | StrOutputParser()
    )
    # Re-asignamos la clave 'input' del historial para que la cadena final la reciba
    final_chain = RunnablePassthrough.assign(
        answer=rag_chain
    )

    chain = final_chain
    logging.info("✅ Arquitectura de IA Experta (v15) inicializada correctamente.")

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
        # La cadena ahora devuelve un diccionario con una sola clave: "answer"
        result = chain.invoke({"input": user_input, "chat_history": current_chat_history})
        answer = result["answer"]

        current_chat_history.extend([
            HumanMessage(content=user_input),
            AIMessage(content=answer)
        ])
        chat_histories[session_id] = current_chat_history[-10:]

        logging.info(f"Respuesta generada para '{session_id}': '{answer}'")
        return jsonify({"text": answer})

    except Exception as e:
        logging.error(f"Error procesando la solicitud RAG: {e}", exc_info=True)
        return jsonify({"text": "Lo siento, ha ocurrido un error al procesar tu solicitud."}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(port=port, host='0.0.0.0', debug=True)