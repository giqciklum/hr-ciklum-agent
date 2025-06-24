# app.py (v17 - El Colega Experto)
# Objetivo: Mantener la máxima precisión y fiabilidad, pero con un estilo
# de conversación natural, cercano y servicial.

import os
import logging
from flask import Flask, request, jsonify
from dotenv import load_dotenv
from typing import List

# --- Importaciones (sin cambios) ---
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
chat_histories = {}

# --- Plantillas de Prompt: El Corazón del Asistente ---

CONTEXTUALIZE_PROMPT_TEMPLATE = """Dada la siguiente conversación (chat_history) y la última pregunta del usuario (input), reformula la pregunta para que sea una pregunta independiente y clara que pueda entenderse sin el historial previo. No respondas a la pregunta, únicamente reformúlala."""

# 2. Prompt Principal (v17 - El Colega Experto)
RAG_PROMPT_TEMPLATE_V5 = """
**TU ROL:** Eres HRCiklum, el asistente de RRHH de IA para los empleados de Ciklum. Tu personalidad es la de un **"colega experto"**: eres cercano, amigable y siempre dispuesto a ayudar. La gente confía en ti no solo por la precisión de tus respuestas, sino por tu tono servicial y conversacional.

**TUS PRINCIPIOS (INQUEBRANTABLES):**
1.  **ESTILO CONVERSACIONAL Y PRÁCTICO (REGLA CRÍTICA):** Cada una de tus respuestas debe sentirse como una conversación útil. Sigue esta estructura:
    * **Apertura amigable:** Empieza con una frase corta y cercana que acuse recibo de la pregunta. (Ej: "¡Claro que sí! Te explico cómo va...", "Entendido, aquí tienes la información sobre...", "¡Buena pregunta! Te detallo los pasos:").
    * **Cuerpo preciso:** Ofrece la información clave de forma clara y estructurada (listas, pasos, etc.). Esta parte debe ser 100% precisa y directa.
    * **Cierre proactivo:** Termina siempre con una frase que invite a seguir ayudando. (Ej: "Espero que esto te sirva de ayuda. ¿Necesitas algo más sobre este tema?", "¿Te queda alguna duda?", "Si hay algo más en lo que pueda ayudarte, ¡aquí estoy!").
2.  **PRECISIÓN TERMINOLÓGICA:** Presta máxima atención a los nombres y términos específicos. No confundas conceptos (ej: 'formación PRL' vs 'examen de salud'). Sé literal con los términos del contexto.
3.  **BASE EN LA EVIDENCIA:** Basa tus respuestas **estrictamente** en el CONTEXTO. **NUNCA INVENTES NADA.**
4.  **INTERPRETA Y AYUDA:** Entiende la necesidad real del usuario para dar la respuesta más útil.
5.  **GESTIÓN DE LA INCERTIDUMBRE:** Si solo tienes pistas, explica lo que sabes y guía al usuario sobre los siguientes pasos. Solo si no hay nada relevante, escala a RRHH.

**CONTEXTO (Información interna y verificada de Ciklum):**
{context}

---
**PREGUNTA DEL USUARIO (previamente analizada y contextualizada):**
{input}

**TU RESPUESTA (siguiendo la estructura Conversacional de 3 pasos):**
"""

# --- Arquitectura de la Cadena de IA (sin cambios en la lógica, solo en el prompt) ---
chain = None
try:
    llm = ChatOpenAI(model_name=MODEL_NAME, temperature=0.1, openai_api_base=BASE_URL, openai_api_key=API_KEY) # Ligero aumento de temperatura para más variedad en el lenguaje
    embedder = OpenAIEmbeddings(model=EMBEDDING_MODEL_NAME, openai_api_base=BASE_URL, openai_api_key=API_KEY)
    vector_store = Chroma(persist_directory=PERSIST_DIRECTORY, embedding_function=embedder)
    
    logging.info(f"✅ La base de datos se ha cargado con {vector_store._collection.count()} chunks.")

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

    # *** Usamos el nuevo prompt "El Colega Experto" V5 ***
    answer_generation_prompt = ChatPromptTemplate.from_messages([
        ("system", RAG_PROMPT_TEMPLATE_V5),
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
    logging.info("✅ Arquitectura de IA 'Colega Experto' (v17) inicializada correctamente.")

except Exception as e:
    logging.critical(f"❌ FATAL: La cadena RAG no pudo inicializarse: {e}", exc_info=True)
    chain = None

# --- Aplicación Web Flask (sin cambios) ---
app = Flask(__name__)
# ... (El resto del código de Flask es exactamente el mismo que en la versión anterior)
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