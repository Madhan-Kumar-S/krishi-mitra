from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.vectorstores import VectorStoreRetriever
from typing import Optional
from langchain.schema import Document
import time
import logging
import random

logger = logging.getLogger(__name__)

def store_embeddings(documents: list[Document], embeddings: GoogleGenerativeAIEmbeddings) -> Optional[Chroma]:
    """
    Store embeddings for the documents using embeddings and Chroma vectorstore.
    Returns the Chroma vectorstore object.
    """
    try:
        # Process documents in smaller batches to avoid quota limits
        batch_size = 5  # Reduced batch size
        vectorstore = None
        max_retries = 3
        base_delay = 2  # Base delay in seconds
        
        for i in range(0, len(documents), batch_size):
            batch = documents[i:i + batch_size]
            retry_count = 0
            
            while retry_count < max_retries:
                try:
                    if vectorstore is None:
                        vectorstore = Chroma.from_documents(
                            documents=batch,
                            embedding=embeddings
                        )
                    else:
                        vectorstore.add_documents(batch)
                    
                    # Add jitter to the delay
                    delay = base_delay + random.uniform(0, 1)
                    time.sleep(delay)
                    break
                    
                except Exception as e:
                    retry_count += 1
                    if "quota" in str(e).lower() or "rate_limit" in str(e).lower():
                        # Exponential backoff with jitter
                        delay = (base_delay * (2 ** retry_count)) + random.uniform(0, 1)
                        logger.warning(f"Rate limit exceeded. Retrying in {delay} seconds...")
                        time.sleep(delay)
                    else:
                        logger.error(f"Error processing batch {i//batch_size + 1}: {str(e)}")
                        raise
            
            if retry_count == max_retries:
                logger.error(f"Max retries exceeded for batch {i//batch_size + 1}")
                raise Exception("Max retries exceeded")
        
        return vectorstore
    except Exception as e:
        logger.error(f"Error creating VectorStoreRetriever from Chroma DB: {e}")
        raise Exception(f"""Error creating VectorStoreRetriever from Chroma DB: {e}""")
