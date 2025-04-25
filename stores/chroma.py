from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.vectorstores import VectorStoreRetriever
from typing import Optional
from langchain.schema import Document
import time
import logging

logger = logging.getLogger(__name__)

def store_embeddings(documents: list[Document], embeddings: GoogleGenerativeAIEmbeddings) -> Optional[Chroma]:
    """
    Store embeddings for the documents using embeddings and Chroma vectorstore.
    Returns the Chroma vectorstore object.
    """
    try:
        # Process documents in smaller batches to avoid quota limits
        batch_size = 10  # Adjust based on your quota limits
        vectorstore = None
        
        for i in range(0, len(documents), batch_size):
            batch = documents[i:i + batch_size]
            try:
                if vectorstore is None:
                    vectorstore = Chroma.from_documents(
                        documents=batch,
                        embedding=embeddings
                    )
                else:
                    vectorstore.add_documents(batch)
                
                # Add delay between batches to respect rate limits
                time.sleep(1)  # Adjust delay as needed
                
            except Exception as e:
                logger.error(f"Error processing batch {i//batch_size + 1}: {str(e)}")
                if "quota" in str(e).lower():
                    # If we hit quota limit, wait longer before retrying
                    time.sleep(5)
                    continue
                raise
        
        return vectorstore
    except Exception as e:
        logger.error(f"Error creating VectorStoreRetriever from Chroma DB: {e}")
        raise Exception(f"""Error creating VectorStoreRetriever from Chroma DB: {e}""")
