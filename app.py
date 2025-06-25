# app.py (Versión 10 - Definitiva, Modular y Optimizada)
import os
import logging
from flask import Flask, request, jsonify
from dotenv import load_dotenv
from typing import List, Dict, Any

from google.cloud import secretmanager

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.documents import Document
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
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
GCP_PROJECT_ID = os.getenv("GCP_PROJECT_ID", "bustling-cosmos-462514-a4")

if not API_KEY or not GCP_PROJECT_ID:
    logging.critical("FATAL: Faltan variables de entorno esenciales (OPENAI_API_KEY o GCP_PROJECT_ID).")
    exit()

# --- "Memoria" del Chatbot ---
chat_histories: Dict[str, Any] = {}

# --- Función para obtener la PERSONALIDAD del prompt desde Secret Manager ---
def get_hr_prompt_personality() -> str:
    try:
        secret_id = "hr-ciklum-prompt"
        version_id = "latest"
        name = f"projects/{GCP_PROJECT_ID}/secrets/{secret_id}/versions/{version_id}"
        client = secretmanager.SecretManagerServiceClient()
        response = client.access_secret_version(request={"name": name})
        prompt_text = response.payload.data.decode("UTF-8")
        logging.info("✅ Personalidad del prompt cargada exitosamente desde Secret Manager.")
        return prompt_text
    except Exception as e:
        logging.error(f"❌ CRITICAL: No se pudo cargar la personalidad del prompt. Usando personalidad de emergencia. Error: {e}")
        # El prompt de emergencia también es solo la personalidad, sin placeholders.
        return """**TU ROL:** Eres HRCiklum... (etc.)"""

# --- Arquitectura de la Cadena de IA ---
final_chain = None
try:
    llm = ChatOpenAI(model_name=MODEL_NAME, temperature=0.0, openai_api_base=BASE_URL, openai_api_key=API_KEY, max_tokens=800, request_timeout=90)
    embedder = OpenAIEmbeddings(model=EMBEDDING_MODEL_NAME, openai_api_base=BASE_URL, openai_api_key=API_KEY)
    vector_store = Chroma(persist_directory=PERSIST_DIRECTORY, embedding_function=embedder)
    logging.info(f"✅ Base de datos cargada con {vector_store._collection.count()} chunks.")

    base_retriever = vector_store.as_retriever(search_kwargs={"k": 10})
    retriever = MultiQueryRetriever.from_llm(retriever=base_retriever, llm=llm)

    contextualize_q_prompt = ChatPromptTemplate.from_messages([
        ("system", """Dada la siguiente conversación (chat_history) y la última pregunta del usuario (input), reformula la pregunta para que sea una pregunta independiente y clara que pueda entenderse sin el historial previo. No respondas a la pregunta, únicamente reformúlala."""),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
    ])
    history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)

    # --- ARQUITECTURA MODULAR Y SEGURA DEL PROMPT ---
    
    # 1. Obtenemos la "personalidad" (el texto limpio) desde Secret Manager UNA VEZ al inicio.
    # Esto lo puede editar un equipo no técnico de forma segura.
    prompt_personality = get_hr_prompt_personality()

    # 2. El código define la "estructura técnica" con los placeholders que LangChain necesita.
    # Esta parte no debe ser tocada por personal no técnico.
    full_system_prompt_template = f"""{prompt_personality}

CONTEXTO (Información interna y verificada de Ciklum):
{{context}}
---
PREGUNTA DEL USUARIO (previamente analizada y contextualizada):
{{input}}
"""
    
    # 3. Se crea la plantilla final combinando ambas partes.
    answer_generation_prompt = ChatPromptTemplate.from_messages([
        ("system", full_system_prompt_template),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
    ])

    document_chain = create_stuff_documents_chain(llm, answer_generation_prompt)

    rag_chain = RunnablePassthrough.assign(
        context=history_aware_retriever
    ).assign(
        answer=document_chain
    )
    
    final_chain = rag_chain | (lambda x: x['answer'])
    
    logging.info("✅ Arquitectura de IA Definitiva (v10 - Modular y Optimizada) inicializada.")

except Exception as e:
    logging.critical(f"❌ FATAL: La cadena RAG no pudo inicializarse: {e}", exc_info=True)


# --- Aplicación Web Flask (Sin cambios) ---
app = Flask(__name__)

# ... (El resto del fichero Flask es idéntico y correcto) ...
@app.route("/chat", methods=["POST"])
def handle_chat_event():
    if not final_chain:
        return jsonify({"text": "Lo siento, el asistente no está disponible en este momento."}), 500
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