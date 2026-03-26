# test_retrieval.py (Migrated to HuggingFace Embeddings)
import sys
import logging
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

PERSIST_DIRECTORY = "chroma_db_v2"

def test_retrieval(question: str):
    """
    Loads the vector database and retrieves the most relevant
    documents for a given question using HuggingFace embeddings.
    """
    logging.info("Loading HuggingFace embeddings (all-MiniLM-L6-v2)...")
    embedder = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    logging.info("Loading existing vector database...")
    try:
        vector_store = Chroma(persist_directory=PERSIST_DIRECTORY, embedding_function=embedder)
        count = vector_store._collection.count()
        logging.info(f"Database loaded with {count} chunks.")

        retriever = vector_store.as_retriever(search_kwargs={"k": 8})

        logging.info(f"\n--- Searching for: '{question}' ---\n")

        relevant_docs = retriever.invoke(question)

        if not relevant_docs:
            logging.warning("No relevant documents were found.")
            return

        for i, doc in enumerate(relevant_docs):
            source = doc.metadata.get('source', 'Unknown')
            page = doc.metadata.get('page', '-')
            content_preview = doc.page_content[:300].replace('\n', ' ') + "..."
            print("----------------------------------------------------------------------")
            print(f"CHUNK #{i+1} | SOURCE: {source} | PAGE: {page}")
            print("----------------------------------------------------------------------")
            print(f"{content_preview}\n")

    except Exception as e:
        logging.error(f"An error occurred during retrieval: {e}", exc_info=True)

if __name__ == "__main__":
    if len(sys.argv) > 1:
        user_question = " ".join(sys.argv[1:])
        test_retrieval(user_question)
    else:
        test_questions = [
            "What is the flexible retribution plan?",
            "¿Cuál es la política de asistencia?",
            "How do I activate my restaurant card?",
        ]
        print("\n=== Running default test queries ===\n")
        for q in test_questions:
            test_retrieval(q)
            print("\n")
