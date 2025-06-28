# app.py (Version Final v6 – Bilingüe, Contextual y con Lógica de Acción Mejorada)
import os
import logging
from flask import Flask, request, jsonify
from dotenv import load_dotenv
from typing import List, Dict, Any

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.documents import Document
from langchain_core.runnables import RunnablePassthrough
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import create_history_aware_retriever
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_core.messages import HumanMessage, AIMessage

# --- Configuración Inicial ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
load_dotenv()

# --- Constantes y Variables de Entorno ---
API_KEY = os.getenv("OPENAI_API_KEY")
BASE_URL = os.getenv("OPENAI_API_BASE", "https://genai-gateway.azure-api.net/")
PERSIST_DIRECTORY = "chroma_db_v2"
MODEL_NAME = "gpt-4o"
EMBEDDING_MODEL_NAME = "text-embedding-3-large"

if not API_KEY:
    logging.critical("FATAL: No se ha encontrado la 'OPENAI_API_KEY'.")
    exit()

# --- "Memoria" del Chatbot ---
chat_histories: Dict[str, Any] = {}

# --- Prompt de Contextualización (sin cambios) ---
CONTEXTUALIZE_PROMPT_TEMPLATE = """Dada la siguiente conversación (chat_history) y la última pregunta del usuario (input), reformula la pregunta para que sea una pregunta independiente y clara que pueda entenderse sin el historial previo. No respondas a la pregunta, únicamente reformúlala."""

# --- PROMPT DEFINITIVO V8 (MODIFICADO) ---
# Esta es la versión mejorada que no cita fuentes y tiene las reglas más claras.
RAG_PROMPT_V8_MODIFICADO = """
**TU ROL:** Eres HR SPAIN CIKLUM BOT, tu Asistente de IA y compañero experto dentro de Ciklum. Tu misión es ser excepcionalmente servicial, proactivo y fiable. No eres un simple buscador de datos, eres un solucionador de problemas.

**TUS PRINCIPIOS (INQUEBRANTABLES):**

1.  **IDIOMA DE RESPUESTA (Regla Maestra):** Detecta el idioma principal de la **PREGUNTA DEL USUARIO** (español o inglés) y responde **siempre** en ese mismo idioma.

2.  **BASE EN LA EVIDENCIA (Regla de Oro):** Basa tus respuestas **única y exclusivamente** en la información del CONTEXTO proporcionado. **NUNCA INVENTES NADA.** Tu conocimiento se limita estrictamente a los documentos internos que te han sido facilitados; no menciones los nombres de los archivos fuente al usuario.

3.  **PENSAMIENTO PASO A PASO (Para Preguntas Complejas):** Ante una pregunta que requiera combinar información de varias fuentes, razona internamente paso a paso (no muestres este razonamiento al usuario):
    * *Paso 1: Identificar las sub-preguntas clave del usuario.*
    * *Paso 2: Localizar la información para cada sub-pregunta en el CONTEXTO proporcionado.*
    * *Paso 3: Sintetizar la información encontrada en una única respuesta coherente, clara y bien estructurada.*

4.  **RESPUESTAS PROACTIVAS COMO PLANES DE ACCIÓN:** No te limites a responder, ¡guía al usuario! Si la pregunta implica una acción (ej. "cómo me doy de alta", "qué formaciones hago"), tu respuesta debe ser un plan de acción claro y numerado. Anticipa la siguiente pregunta del usuario y añade información útil relacionada.

5.  **DISCRIMINACIÓN PRECISA:** El contexto puede contener información sobre varios procesos similares. Si el usuario pregunta específicamente por un proceso, **enfoca tu respuesta exclusivamente en ese proceso** para evitar confusiones.

6.  **GESTIÓN DE INCERTIDUMBRE Y RESILIENCIA (Protocolo Mejorado):**
    * **Si el contexto es pobre pero relevante:** No te rindas inmediatamente. Intenta dar una respuesta parcial y útil. Por ejemplo: "No he encontrado el correo de esa persona, pero te confirmo que su rol es Delivery Coordinator. Puedes buscarlo en el directorio de la empresa o preguntar a tu manager."
    * **Si el contexto está vacío o no es relevante:** Responde con amabilidad (en el idioma del usuario): "He revisado la documentación interna, pero no he encontrado una respuesta directa a tu pregunta. Para darte la información más precisa, te recomiendo consultarlo con el equipo de RRHH. Puedes explicarles tu caso así: '[Sugerencia de cómo el usuario puede formular la pregunta a RRHH]'."
    * **Si la pregunta es sobre leyes externas o temas legales complejos:** Explica tu función: "Mi conocimiento se basa en las políticas internas de Ciklum. Para interpretaciones de leyes externas o asuntos legales, el equipo de RRHH es el contacto adecuado para darte una orientación precisa."

7.  **TONO AMIGABLE Y COMPAÑERO:** Usa un tono cercano, positivo y empático. Eres un compañero más del equipo. Termina siempre tus respuestas con una frase de ayuda como "Espero que esto te sea de gran ayuda" o "Si algo no queda claro, dímelo y lo vemos de otra forma".

**CONTEXTO (Información interna y verificada de Ciklum):**
{context}
---
**PREGUNTA DEL USUARIO (previamente analizada y contextualizada):**
{input}
**TU RESPUESTA (proactiva, estructurada y en el mismo idioma que la pregunta):**
"""

# --- Arquitectura de la Cadena de IA ---
final_chain = None
try:
    llm = ChatOpenAI(
        model_name=MODEL_NAME,
        temperature=0.0,
        openai_api_base=BASE_URL,
        openai_api_key=API_KEY,
        max_tokens=800,
        request_timeout=90
    )
    embedder = OpenAIEmbeddings(model=EMBEDDING_MODEL_NAME, openai_api_base=BASE_URL, openai_api_key=API_KEY)
    vector_store = Chroma(persist_directory=PERSIST_DIRECTORY, embedding_function=embedder)
    logging.info(f"✅ Base de datos cargada con {vector_store._collection.count()} chunks.")

    base_retriever = vector_store.as_retriever(search_kwargs={"k": 10})
    retriever = MultiQueryRetriever.from_llm(retriever=base_retriever, llm=llm)

    def format_docs(docs: List[Document]) -> str:
        if not docs:
            logging.warning("El retriever no ha devuelto ningún documento.")
            return ""
        logging.info(f"Retriever ha encontrado {len(docs)} documentos para el contexto.")
        return "\n\n".join(doc.page_content for doc in docs)

    # --- Creación de la cadena principal ---
    contextualize_q_prompt = ChatPromptTemplate.from_messages([
        ("system", CONTEXTUALIZE_PROMPT_TEMPLATE),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
    ])
    history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)

    answer_generation_prompt = ChatPromptTemplate.from_messages([
        # Usamos el prompt definitivo V8 Modificado
        ("system", RAG_PROMPT_V8_MODIFICADO),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
    ])

    answer_chain = create_stuff_documents_chain(llm, answer_generation_prompt)

    rag_chain = RunnablePassthrough.assign(
        context=history_aware_retriever,
    ).assign(
        answer=answer_chain,
    )

    final_chain = rag_chain | (lambda x: x['answer'])
    
    logging.info("✅ Arquitectura de IA Experta (v6 - Lógica Mejorada) inicializada correctamente.")

except Exception as e:
    logging.critical(f"❌ FATAL: La cadena RAG no pudo inicializarse: {e}", exc_info=True)


# --- Aplicación Web Flask ---
app = Flask(__name__)

@app.route("/chat", methods=["POST"])
def handle_chat_event():
    if not final_chain:
        return jsonify({"text": "Lo siento, el asistente no está disponible en este momento. Por favor, revisa los logs del servidor."}), 500

    data = request.json
    user_input = data.get('message', {}).get('text', '').strip()
    session_id = data.get('user', {}).get('id', 'default_session')

    if not user_input or data.get('user', {}).get('type') == 'BOT':
        return jsonify({})

    logging.info(f"Consulta recibida de '{session_id}': '{user_input}'")
    
    current_chat_history = chat_histories.get(session_id, [])
    
    try:
        result = final_chain.invoke({
            "input": user_input,
            "chat_history": current_chat_history
        })
        answer_for_user = result

        current_chat_history.extend([
            HumanMessage(content=user_input),
            AIMessage(content=answer_for_user)
        ])
        chat_histories[session_id] = current_chat_history[-10:] # Keep last 5 pairs of messages

        logging.info(f"Respuesta generada para '{session_id}': '{answer_for_user}'")
        return jsonify({"text": answer_for_user})

    except Exception as e:
        logging.error(f"Error procesando la solicitud RAG: {e}", exc_info=True)
        return jsonify({"text": "Lo siento, ha ocurrido un error al procesar tu solicitud."}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(port=port, host='0.0.0.0', debug=True)