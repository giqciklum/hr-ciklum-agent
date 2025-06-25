# app.py (Versión Final v5 - Bilingüe, Contextual y con Prompt Dinámico desde Secret Manager)
import os
import logging
from flask import Flask, request, jsonify
from dotenv import load_dotenv
from typing import List, Dict, Any

# IMPORTACIÓN CLAVE PARA SECRET MANAGER
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

# ID de tu proyecto de Google Cloud
GCP_PROJECT_ID = "619993214373"

if not API_KEY:
    logging.critical("FATAL: No se ha encontrado la 'OPENAI_API_KEY'.")
    exit()

# --- "Memoria" del Chatbot ---
chat_histories: Dict[str, Any] = {}

# --- Prompt de Contextualización ---
CONTEXTUALIZE_PROMPT_TEMPLATE = """Dada la siguiente conversación (chat_history) y la última pregunta del usuario (input), reformula la pregunta para que sea una pregunta independiente y clara que pueda entenderse sin el historial previo. No respondas a la pregunta, únicamente reformúlala."""

# --- NUEVA FUNCIÓN PARA OBTENER EL PROMPT DESDE SECRET MANAGER ---
def get_hr_prompt() -> str:
    """
    Obtiene la última versión del prompt de HR desde Google Cloud Secret Manager.
    Si falla, devuelve un prompt de emergencia para que la app no se caiga.
    """
    try:
        secret_id = "hr-ciklum-prompt"
        version_id = "latest"
        name = f"projects/{GCP_PROJECT_ID}/secrets/{secret_id}/versions/{version_id}"

        client = secretmanager.SecretManagerServiceClient()
        response = client.access_secret_version(request={"name": name})
        
        prompt_text = response.payload.data.decode("UTF-8")
        logging.info("✅ Prompt cargado exitosamente desde Secret Manager.")
        return prompt_text

    except Exception as e:
        logging.error(f"❌ CRITICAL: No se pudo cargar el prompt desde Secret Manager. Usando prompt de emergencia. Error: {e}", exc_info=True)
        # Este es un prompt de respaldo para que la app siga funcionando si Secret Manager falla
        return """Eres un asistente de RRHH. Responde de forma profesional y muy breve a las preguntas del usuario.
        Informa al usuario que estás operando en modo de contingencia.
        Contexto: {context}
        Pregunta: {input}
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

    # AQUÍ ESTÁ LA MAGIA: LLAMAMOS A LA FUNCIÓN PARA OBTENER EL PROMPT
    answer_generation_prompt = ChatPromptTemplate.from_messages([
        ("system", get_hr_prompt()),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
    ])

    document_chain = create_stuff_documents_chain(llm, answer_generation_prompt)

    rag_chain = RunnablePassthrough.assign(
        context=history_aware_retriever | format_docs
    ).assign(
        answer=document_chain
    )
    
    # La salida final es solo la clave 'answer'
    final_chain = rag_chain | (lambda x: x['answer'])
    
    logging.info("✅ Arquitectura de IA Experta (v5 - Prompt Dinámico) inicializada correctamente.")

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
