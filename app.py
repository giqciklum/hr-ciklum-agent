# app.py (Corregido y Funcional)
import os
import logging
from flask import Flask, request, jsonify
from dotenv import load_dotenv
from typing import List, Dict, Any
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.documents import Document
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import create_history_aware_retriever
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import (
    LLMChainExtractor,
    EmbeddingsFilter,
    DocumentCompressorPipeline # <-- CORRECCIÓN: Importación añadida
)
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
4.  **GESTIÓN DE INCERTIDUMBRE:** Si el CONTEXTO es parcial, di lo que sabes y guía al siguiente paso. Solo si el CONTEXTO está vacío, responde: "He revisado la documentación interna y no he encontrado información sobre este tema. Para una respuesta precisa, consulta directamente con RRHH."
5.  **CLARIDAD Y CONCISIÓN:** Responde en **menos de 100 palabras** o un máximo de 5 puntos clave. Ve al grano.
6.  **SIN RELLENO:** Evita frases como "Espero que esto ayude" o "Como modelo de IA...". Sé directo y profesional.
**CONTEXTO (Información interna y verificada de Ciklum):**
{context}
---
**PREGUNTA DEL USUARIO (previamente analizada y contextualizada):**
{input}
**TU RESPUESTA (clara, práctica y basada 100% en el contexto):**
"""
SHORTEN_PROMPT = ChatPromptTemplate.from_template(
    "Resume la siguiente respuesta de un asistente de RRHH en menos de 100 palabras, manteniendo la terminología y los puntos clave. Sé directo y práctico.\n\nRESPUESTA ORIGINAL:\n{rag_answer}"
)

# --- Arquitectura de la Cadena de IA ---
final_chain = None
try:
    llm = ChatOpenAI(
        model_name=MODEL_NAME,
        temperature=0.0,
        openai_api_base=BASE_URL,
        openai_api_key=API_KEY,
        max_tokens=350,
        request_timeout=60
    )
    embedder = OpenAIEmbeddings(model=EMBEDDING_MODEL_NAME, openai_api_base=BASE_URL, openai_api_key=API_KEY)
    vector_store = Chroma(persist_directory=PERSIST_DIRECTORY, embedding_function=embedder)
    logging.info(f"✅ Base de datos cargada con {vector_store._collection.count()} chunks.")

    # === CORRECCIÓN: Se usa DocumentCompressorPipeline para encadenar los compresores ===
    base_retriever = vector_store.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 12, "lambda_mult": 0.5, "fetch_k": 25}
    )
    embeddings_filter = EmbeddingsFilter(embeddings=embedder, similarity_threshold=0.78)
    llm_extractor = LLMChainExtractor.from_llm(llm)
    
    pipeline_compressor = DocumentCompressorPipeline(
        transformers=[embeddings_filter, llm_extractor]
    )
    
    contextual_compression_retriever = ContextualCompressionRetriever(
        base_compressor=pipeline_compressor,
        base_retriever=base_retriever
    )
    # === FIN DE LA CORRECCIÓN ===

    def format_docs(docs: List[Document]) -> str:
        return "\n\n".join(doc.page_content for doc in docs)

    contextualize_q_prompt = ChatPromptTemplate.from_messages([
        ("system", CONTEXTUALIZE_PROMPT_TEMPLATE),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
    ])
    history_aware_retriever = create_history_aware_retriever(llm, contextual_compression_retriever, contextualize_q_prompt)
    
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
    
    conciseness_chain = SHORTEN_PROMPT | llm | StrOutputParser()

    def add_postamble(answer: str) -> str:
        postamble = "\n\n*Respuesta generada por HRCiklum. Escribe 'detalles' si necesitas más información.*"
        return answer.strip() + postamble

    # === CORRECCIÓN: La cadena final devuelve un diccionario para poder usar la función "detalles" ===
    final_chain = {
        "rag_answer": rag_chain,
    } | RunnablePassthrough.assign(
        # Descomenta la siguiente línea para activar la capa de concisión.
        # answer=lambda x: conciseness_chain.invoke({"rag_answer": x["rag_answer"]})
        # O usa esta línea para la respuesta directa de RAG (más rápido).
        answer=RunnableLambda(lambda x: x["rag_answer"])
    )
    # La cadena ahora devuelve un diccionario: {'rag_answer': '...', 'answer': '...'}
    
    logging.info("✅ Arquitectura de IA Experta (v16) inicializada correctamente.")

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
    
    # === CORRECCIÓN: Lógica para "detalles" y manejo del historial ===
    current_chat_history = chat_histories.get(session_id, [])
    
    try:
        if user_input.lower() == "detalles" and chat_histories.get(f"{session_id}_last_rag_answer"):
            # Si el usuario pide detalles, le damos la última respuesta larga guardada
            answer_for_user = chat_histories[f"{session_id}_last_rag_answer"]
        else:
            # Invoca la cadena principal
            result = final_chain.invoke({
                "input": user_input,
                "chat_history": current_chat_history
            })
            
            # Guardamos la respuesta larga por si la piden después
            chat_histories[f"{session_id}_last_rag_answer"] = result['rag_answer']
            
            # La respuesta para el usuario es la versión corta con el post-amble
            answer_for_user = add_postamble(result['answer'])

        # Actualizar el historial de conversación para la siguiente pregunta
        current_chat_history.extend([
            HumanMessage(content=user_input),
            AIMessage(content=answer_for_user)
        ])
        chat_histories[session_id] = current_chat_history[-10:] # Guardar solo los últimos 5 turnos

        logging.info(f"Respuesta generada para '{session_id}': '{answer_for_user}'")
        return jsonify({"text": answer_for_user})

    except Exception as e:
        logging.error(f"Error procesando la solicitud RAG: {e}", exc_info=True)
        return jsonify({"text": "Lo siento, ha ocurrido un error al procesar tu solicitud."}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(port=port, host='0.0.0.0', debug=True)