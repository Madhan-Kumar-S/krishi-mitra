from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.vectorstores import VectorStoreRetriever
from typing import Optional
from langchain.schema import Document
import logging

logger = logging.getLogger(__name__)

def store_embeddings(documents: list[Document], embeddings=None) -> Optional[Chroma]:
    """
    Store embeddings for the documents using HuggingFace embeddings and Chroma vectorstore.
    Returns the Chroma vectorstore object.
    """
    try:
        # Initialize HuggingFace embeddings if not provided
        if embeddings is None:
            embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2",  # Lightweight model
                model_kwargs={'device': 'cpu'},
                encode_kwargs={'normalize_embeddings': True}
            )

        # Create vectorstore with all documents at once
        vectorstore = Chroma.from_documents(
            documents=documents,
            embedding=embeddings
        )
        
        return vectorstore
    except Exception as e:
        logger.error(f"Error creating VectorStoreRetriever from Chroma DB: {e}")
        raise Exception(f"""Error creating VectorStoreRetriever from Chroma DB: {e}""")
