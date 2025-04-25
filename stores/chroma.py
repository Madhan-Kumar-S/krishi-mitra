from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.vectorstores import VectorStoreRetriever
from typing import Optional
from langchain.schema import Document


def store_embeddings(documents: list[Document], embeddings: GoogleGenerativeAIEmbeddings) -> Optional[Chroma]:
    """
    Store embeddings for the documents using embeddings and Chroma vectorstore.
    Returns the Chroma vectorstore object.
    """
    try:
        vectorstore_web = Chroma.from_documents(
            documents=documents,
            embedding=embeddings
            #persist_directory="./chroma_data"
            # Removed persist_directory to use in-memory storage
        )
        return vectorstore_web
    except Exception as e:
        raise Exception(f"""Error creating VectorStoreRetriever from Chroma DB: {e}""")
