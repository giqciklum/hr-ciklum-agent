# app.py (con Control de Calidad)
import os
import logging
from flask import Flask, request, jsonify
from dotenv import load_dotenv
from typing import List, Dict, Any
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.documents import Document
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import create_history_aware_retriever
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

# --- Plantillas de Prompt ---
CONTEXTUALIZE_PROMPT_TEMPLATE = """Dada la siguiente conversación (chat_history) y la última pregunta del usuario (input), reformula la pregunta para que sea una pregunta independiente y clara que pueda entenderse sin el historial previo. No respondas a la pregunta, únicamente reformúlala."""
RAG_PROMPT_V4 = """
**TU ROL:** Eres HRCiklum, el asistente de RRHH de IA para empleados de Ciklum. Eres un compañero experto, fiable y práctico. Tu credibilidad es tu mayor valor.
**TUS PRINCIPIOS (INQUEBRANTABLES):**
1.  **BASE EN LA EVIDENCIA:** Basa tus respuestas **única y exclusivamente** en la información del CONTEXTO proporcionado. **NUNCA INVENTES NADA.**
2.  **INTERPRETA Y AYUDA:** Infiere la necesidad real del usuario. Si preguntan "¿quién me ayuda?", busca en el CONTEXTO el contacto o procedimiento específico para su problema implícito.
3.  **RESPUESTAS PRÁCTICAS:** Comunícate de forma natural. Usa listas, puntos clave o pasos a seguir. El objetivo es que el empleado sepa exactamente qué hacer.
4.  **GESTIÓN DE INCERTIDUMBRE:** Si el CONTEXTO está vacío o no es relevante, responde: "He revisado la documentación interna y no he encontrado información sobre este tema. Para una respuesta precisa, consulta directamente con RRHH."
5.  **CLARIDAD Y CONCISIÓN:** Responde de forma directa y profesional.

**CONTEXTO (Información interna y verificada de Ciklum):**
{context}
---
**PREGUNTA DEL USUARIO (previamente analizada y contextualizada):**
{input}
**TU RESPUESTA (clara, práctica y basada 100% en el contexto):**
"""

# --- Arquitectura de la Cadena de IA ---
final_chain = None
try:
    llm = ChatOpenAI(
        model_name=MODEL_NAME,
        temperature=0.0,
        openai_api_base=BASE_URL,
        openai_api_key=API_KEY,
        max_tokens=400,
        request_timeout=90
    )
    embedder = OpenAIEmbeddings(model=EMBEDDING_MODEL_NAME, openai_api_base=BASE_URL, openai_api_key=API_KEY)
    vector_store = Chroma(persist_directory=PERSIST_DIRECTORY, embedding_function=embedder)
    logging.info(f"✅ Base de datos cargada con {vector_store._collection.count()} chunks.")

    # === CAMBIO CLAVE: Retriever con control de calidad ===
    # Busca hasta 8 documentos, pero filtra los que no tengan una mínima relevancia (score > 0.4)
    # Esto evita que documentos totalmente irrelevantes ensucien el contexto.
    base_retriever = vector_store.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={"k": 8, "score_threshold": 0.4}
    )
    # === FIN DEL CAMBIO ===
    
    def format_docs(docs: List[Document]) -> str:
        if not docs:
            logging.warning("El retriever no ha devuelto ningún documento tras el filtrado por score.")
            return ""
        
        logging.info(f"Retriever ha encontrado {len(docs)} documentos relevantes para el contexto.")
        for i, doc in enumerate(docs):
            logging.info(f"  - Doc {i+1} (Fuente: {doc.metadata.get('source', 'N/A')}): {doc.page_content[:100]}...")
            
        return "\n\n".join(doc.page_content for doc in docs)

    contextualize_q_prompt = ChatPromptTemplate.from_messages([
        ("system", CONTEXTUALIZE_PROMPT_TEMPLATE),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
    ])
    
    history_aware_retriever = create_history_aware_retriever(llm, base_retriever, contextualize_q_prompt)
    
    answer_generation_prompt = ChatPromptTemplate.from_messages([
        ("system", RAG_PROMPT_V4),
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

    final_chain = rag_chain
    logging.info("✅ Arquitectura de IA Experta (v18 - Calidad y Robustez) inicializada correctamente.")

except Exception as e:
    logging.critical(f"❌ FATAL: La cadena RAG no pudo inicializarse: {e}", exc_info=True)

# (El resto del fichero Flask se mantiene igual)
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
