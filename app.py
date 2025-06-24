# app.py (v16 - El Experto Meticuloso)
# Objetivo: Eliminar la confusión entre conceptos relacionados mediante un prompt
# ultra-preciso y un retriever más selectivo.

import os
import logging
from flask import Flask, request, jsonify
from dotenv import load_dotenv
from typing import List

# --- Importaciones Clave (sin cambios) ---
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

# --- Configuración Inicial y Constantes (sin cambios) ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
load_dotenv()
API_KEY = os.getenv("OPENAI_API_KEY")
BASE_URL = os.getenv("OPENAI_API_BASE", "https://genai-gateway.azure-api.net/")
PERSIST_DIRECTORY = "chroma_db"
MODEL_NAME = "gpt-4o"
EMBEDDING_MODEL_NAME = "text-embedding-3-large"
chat_histories = {} # "Memoria" del Chatbot

# --- Plantillas de Prompt: El Corazón del Asistente ---

CONTEXTUALIZE_PROMPT_TEMPLATE = """Dada la siguiente conversación (chat_history) y la última pregunta del usuario (input), reformula la pregunta para que sea una pregunta independiente y clara que pueda entenderse sin el historial previo. No respondas a la pregunta, únicamente reformúlala."""

# 2. Prompt Principal (v16 - El Experto Meticuloso)
RAG_PROMPT_TEMPLATE_V4 = """
**TU ROL:** Eres HRCiklum, el asistente de RRHH de IA para los empleados de Ciklum. Tu identidad es la de un compañero experto: eres la persona a la que todos acuden porque das respuestas fiables, prácticas y, sobre todo, precisas. Tu reputación se basa en tu meticulosidad.

**TUS PRINCIPIOS (INQUEBRANTABLES):**
1.  **BASE EN LA EVIDENCIA:** Tus respuestas se basan **única y exclusivamente** en la información del CONTEXTO proporcionado. **NUNCA INVENTES NADA.**
2.  **PRECISIÓN TERMINOLÓGICA (REGLA CRÍTICA):** Presta máxima atención a los nombres y términos específicos. **No confundas conceptos relacionados pero distintos.** Por ejemplo, 'formación en Prevención de Riesgos Laborales' es un curso online y es **diferente** de 'examen de salud' que es una cita médica. Si el usuario pregunta por el concepto A, responde **exactamente** sobre A, incluso si la información sobre un concepto B similar está en el mismo fragmento de contexto. Sé literal con los términos.
3.  **INTERPRETA Y AYUDA:** Entiende la necesidad real del usuario. Si preguntan "¿quién me ayuda?", infiere el tema y busca el contacto o procedimiento correcto.
4.  **RESPUESTAS PRÁCTICAS Y DIRECTAS:** Comunícate de forma natural. Estructura la información con listas o pasos a seguir para que sea fácil de entender y accionar.
5.  **GESTIÓN DE LA INCERTIDUMBRE:** Si el contexto solo ofrece pistas, explica lo que has encontrado y guía al usuario. Solo si no hay nada en absoluto, escala a RRHH.

**CONTEXTO (Información interna y verificada de Ciklum):**
{context}

---
**PREGUNTA DEL USUARIO (previamente analizada y contextualizada):**
{input}

**TU RESPUESTA (precisa, práctica y basada 100% en el contexto):**
"""

# --- Arquitectura de la Cadena de IA ---
chain = None
try:
    llm = ChatOpenAI(model_name=MODEL_NAME, temperature=0.0, openai_api_base=BASE_URL, openai_api_key=API_KEY)
    embedder = OpenAIEmbeddings(model=EMBEDDING_MODEL_NAME, openai_api_base=BASE_URL, openai_api_key=API_KEY)
    vector_store = Chroma(persist_directory=PERSIST_DIRECTORY, embedding_function=embedder)

    item_count = vector_store._collection.count()
    logging.info(f"✅ La base de datos se ha cargado con {item_count} chunks.")
    if item_count == 0:
        logging.warning("ADVERTENCIA: La base de datos está vacía. Ejecuta build_index.py con la hiper-fragmentación.")

    # *** CAMBIO: RETRIEVER MÁS SELECTIVO ***
    # Reducimos k a 8 para ser más exigentes en la búsqueda inicial.
    base_retriever = vector_store.as_retriever(search_kwargs={"k": 8})
    
    document_compressor = LLMChainExtractor.from_llm(llm)
    contextual_compression_retriever = ContextualCompressionRetriever(
        base_compressor=document_compressor,
        base_retriever=base_retriever
    )

    def format_docs(docs: List[Document]) -> str:
        return "\n\n".join(doc.page_content for doc in docs)

    contextualize_q_prompt = ChatPromptTemplate.from_messages([
        ("system", CONTEXTUALIZE_PROMPT_TEMPLATE),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
    ])
    
    history_aware_retriever = create_history_aware_retriever(llm, contextual_compression_retriever, contextualize_q_prompt)

    # *** CAMBIO: USAMOS EL NUEVO PROMPT "METICULOSO" V4 ***
    answer_generation_prompt = ChatPromptTemplate.from_messages([
        ("system", RAG_PROMPT_TEMPLATE_V4),
        ("human", "{input}"),
    ])

    rag_chain = (
        {
            "context": history_aware_retriever | format_docs, 
            "input": RunnablePassthrough()
        }
        | answer_generation_prompt
        | llm
        | StrOutputParser()
    )
    
    final_chain = RunnablePassthrough.assign(answer=rag_chain)
    chain = final_chain
    logging.info("✅ Arquitectura de IA Meticulosa (v16) inicializada correctamente.")

except Exception as e:
    logging.critical(f"❌ FATAL: La cadena RAG no pudo inicializarse: {e}", exc_info=True)
    chain = None

# --- Aplicación Web Flask (sin cambios) ---
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

        logging.info(f"Respuesta generada para '{session_id}': '{answer}'")
        return jsonify({"text": answer})

    except Exception as e:
        logging.error(f"Error procesando la solicitud RAG: {e}", exc_info=True)
        return jsonify({"text": "Lo siento, ha ocurrido un error al procesar tu solicitud."}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(port=port, host='0.0.0.0', debug=True)