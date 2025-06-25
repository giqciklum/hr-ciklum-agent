# app.py (Versión Mejorada - Experta y Fiable)
import os
import logging
from flask import Flask, request, jsonify
from dotenv import load_dotenv
from typing import List, Dict, Any

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.documents import Document
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import create_history_aware_retriever
from langchain.chains.combine_documents import create_stuff_documents_chain
# ¡NUEVA IMPORTACIÓN!
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

# --- CAMBIO 1: RAG_PROMPT_V5 - EL PROMPT EVOLUCIONADO ---
CONTEXTUALIZE_PROMPT_TEMPLATE = """Dada la siguiente conversación (chat_history) y la última pregunta del usuario (input), reformula la pregunta para que sea una pregunta independiente y clara que pueda entenderse sin el historial previo. No respondas a la pregunta, únicamente reformúlala."""

RAG_PROMPT_V5 = """
**TU ROL:** Eres HRCiklum, el asistente de IA y compañero de confianza para los empleados de Ciklum. Tu objetivo es proporcionar respuestas claras, fiables y prácticas, actuando como un miembro experto y servicial del equipo de RRHH.

**TUS PRINCIPIOS (INQUEBRANTABLES):**
1.  **BASE EN LA EVIDENCIA (Regla de Oro):** Basa tus respuestas **única y exclusivamente** en la información del CONTEXTO proporcionado. **NUNCA INVENTES NADA.** Si un detalle no está en el contexto, no lo menciones.
2.  **SÍNTESIS EXPERTA:** La pregunta del usuario puede ser compleja y la respuesta puede estar repartida en varios fragmentos del contexto. Tu tarea es **sintetizar toda la información relevante** en una única respuesta coherente y bien estructurada.
3.  **RESPUESTAS PRÁCTICAS Y SERVICIALES:** Ve al grano. Usa listas, negritas y pasos a seguir para que el empleado sepa exactamente qué hacer. Anticipa la necesidad real: si preguntan por un "problema", responde con la "solución" que se encuentra en el contexto.
4.  **GESTIÓN DE INCERTIDUMBRE (Protocolo Mejorado):**
    * Si el CONTEXTO está vacío o claramente no es relevante para la pregunta, responde con amabilidad: "He revisado la documentación interna, pero no he encontrado información específica sobre este tema. Para asegurar que recibes una respuesta precisa, lo mejor es que consultes directamente con el equipo de RRHH. ¡Están para ayudarte!"
    * Si el usuario pregunta sobre leyes externas (p.ej. "Estatuto de los Trabajadores") o pide comparaciones que no están en el contexto, explica tu función: "Mi conocimiento se basa en las políticas internas de Ciklum. Para interpretaciones de leyes externas o asuntos legales, el equipo de RRHH es el contacto adecuado para darte una orientación precisa."
5.  **TONO AMIGABLE Y PROFESIONAL:** Sé cercano y servicial, pero siempre preciso y fiable. Termina tus respuestas con una nota positiva o una frase de ayuda como "Espero que esto te sea de ayuda" o "Si tienes otra duda, aquí estoy para ayudarte".

**CONTEXTO (Información interna y verificada de Ciklum):**
{context}
---
**PREGUNTA DEL USUARIO (previamente analizada y contextualizada):**
{input}
**TU RESPUESTA (clara, práctica, servicial y basada 100% en el contexto):**
"""

# --- Arquitectura de la Cadena de IA ---
final_chain = None
try:
    llm = ChatOpenAI(
        model_name=MODEL_NAME,
        temperature=0.0,
        openai_api_base=BASE_URL,
        openai_api_key=API_KEY,
        max_tokens=800, # Aumentamos ligeramente para respuestas más completas
        request_timeout=90
    )
    embedder = OpenAIEmbeddings(model=EMBEDDING_MODEL_NAME, openai_api_base=BASE_URL, openai_api_key=API_KEY)
    vector_store = Chroma(persist_directory=PERSIST_DIRECTORY, embedding_function=embedder)
    logging.info(f"✅ Base de datos cargada con {vector_store._collection.count()} chunks.")

    # === CAMBIO 2: REVOLUCIÓN DEL RETRIEVER -> MultiQueryRetriever ===
    # En lugar de una búsqueda simple, usamos el LLM para generar varias preguntas
    # y buscar documentos para todas ellas. Esto es clave para las preguntas complejas.
    base_retriever = vector_store.as_retriever(search_kwargs={"k": 10}) # Aumentamos k para tener más documentos potenciales
    
    retriever = MultiQueryRetriever.from_llm(
        retriever=base_retriever,
        llm=llm
    )
    
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
    
    # --- CAMBIO 3: CADENA DE GENERACIÓN DE RESPUESTA MÁS CLARA ---
    # Usamos `create_stuff_documents_chain` que está optimizado para esto.
    answer_generation_prompt = ChatPromptTemplate.from_messages([
        ("system", RAG_PROMPT_V5),
        MessagesPlaceholder(variable_name="chat_history"), # Incluimos el historial por si es útil
        ("human", "{input}"),
    ])

    Youtube_chain = create_stuff_documents_chain(llm, answer_generation_prompt)

    # La cadena final ahora combina la recuperación consciente del historial con la generación de respuestas.
    rag_chain = RunnablePassthrough.assign(
        context=history_aware_retriever,
    ).assign(
        answer=Youtube_chain,
    )

    # Extraemos solo la respuesta final para el usuario.
    final_chain = rag_chain | (lambda x: x['answer'])
    
    logging.info("✅ Arquitectura de IA Experta (v2 - MultiQuery) inicializada correctamente.")

except Exception as e:
    logging.critical(f"❌ FATAL: La cadena RAG no pudo inicializarse: {e}", exc_info=True)


# --- Aplicación Web Flask (Sin cambios aquí) ---
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
        chat_histories[session_id] = current_chat_history[-10:]

        logging.info(f"Respuesta generada para '{session_id}': '{answer_for_user}'")
        return jsonify({"text": answer_for_user})

    except Exception as e:
        logging.error(f"Error procesando la solicitud RAG: {e}", exc_info=True)
        return jsonify({"text": "Lo siento, ha ocurrido un error al procesar tu solicitud."}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(port=port, host='0.0.0.0', debug=True)