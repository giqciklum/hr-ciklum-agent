# test_retrieval.py (Corrected Version)
import os # <-- This was the missing line
import sys
import logging
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

# Load environment variables
load_dotenv()
API_KEY = os.getenv("YOUR_API_KEY")
BASE_URL = "https://your-api-gateway-url/"
PERSIST_DIRECTORY = "chroma_db"

def test_retrieval(question: str):
    """
    This function loads the vector database and retrieves the
    most relevant documents for a given question.
    """
    if not API_KEY:
        logging.error("YOUR_API_KEY not found. Make sure your .env file is correct.")
        return

    logging.info("Loading existing vector database...")
    try:
        embedder = OpenAIEmbeddings(
            model="text-embedding-3-large",
            openai_api_base=BASE_URL,
            openai_api_key=API_KEY
        )
        vector_store = Chroma(persist_directory=PERSIST_DIRECTORY, embedding_function=embedder)
        retriever = vector_store.as_retriever(search_kwargs={"k": 8})
        
        logging.info(f"\n--- Searching for relevant documents for question: '{question}' ---\n")
        
        # Retrieve the documents
        relevant_docs = retriever.get_relevant_documents(question)
        
        if not relevant_docs:
            logging.warning("No relevant documents were found.")
            return

        # Print the results
        for i, doc in enumerate(relevant_docs):
            source = doc.metadata.get('source', 'Unknown')
            content_preview = doc.page_content.replace('\n', ' ') + "..."
            print("----------------------------------------------------------------------")
            print(f"CHUNK #{i+1} | SOURCE: {source}")
            print("----------------------------------------------------------------------")
            print(f"{content_preview}\n")

    except Exception as e:
        logging.error(f"An error occurred during retrieval: {e}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        user_question = " ".join(sys.argv[1:])
        test_retrieval(user_question)
    else:
        print("\nPlease provide a question to test.")
        print("Example Usage:")
        print("python test_retrieval.py \"what are the economic consequences of the relocation policy\"")