from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import logging
from llm_setup.llm_setup import LLMService
import configs.config as config
import processing.documents as document_processing
from stores.chroma import store_embeddings
import speech_to_text.gemini as gemini
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import tempfile

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:19006",  # React Native development server
        "http://localhost:19000",  # React Native development server alternative
        "exp://localhost:19000",   # Expo development
        "exp://localhost:19006",   # Expo development alternative
        "https://your-production-app-url.com"  # Replace with your production app URL
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Set up logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Set environment variables
config.set_envs()

# Load documents and store embeddings
json_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "myschemes_scraped.json")
documents = document_processing.load_json_to_langchain_document_schema(json_file_path)
chroma = store_embeddings(documents, config.EMBEDDINGS)
retriever = chroma.as_retriever()

class QueryRequest(BaseModel):
    text: str

class AudioQueryRequest(BaseModel):
    audio_url: str

class Language(BaseModel):
    text: str
    language: str
    language_code: str

# Initialize the LLMService
llm_svc = LLMService(logger, "", retriever)
if llm_svc.error:
    logger.error(f"Error initializing LLM service: {llm_svc.error}")

llm = llm_svc.get_llm()

# Set up the JSON output parser and prompt template
parser = JsonOutputParser(pydantic_object=Language)
prompt_template = PromptTemplate(
    template="Answer the user query.\n{format_instructions}\n{query}\n",
    input_variables=["query"],
    partial_variables={"format_instructions": parser.get_format_instructions()},
)

json_chain = prompt_template | llm | parser

def hallucination_score(context, answer):
    vectorizer = TfidfVectorizer().fit_transform([context, answer])
    vectors = vectorizer.toarray()
    score = cosine_similarity([vectors[0]], [vectors[1]])[0][0]
    return score

@app.post("/query")
async def process_query(request: QueryRequest):
    try:
        llm_response = json_chain.invoke({
            "query": f"""
            User Input: {request.text}
            Provide a JSON response with the following keys:
            * language: The language of the input text.
            * text: A proper English translation understandable by a native English speaker.
            * language_code: The equivalent Google Cloud Platform language code for text-to-speech.
            """
        })
        response_dict = llm_response

        user_language = response_dict['language']
        user_input = response_dict['text']
        user_language_code = response_dict["language_code"]

        docs_with_scores = chroma.similarity_search_with_score(user_input)
        top_docs = docs_with_scores[:4]
        context = " ".join(doc.page_content for doc, _ in top_docs)

        if not context:
            raise HTTPException(status_code=404, detail="No relevant information found")

        prompt = f"""
        You are a highly knowledgeable assistant specializing in Indian government schemes. 
        Your task is to provide clear, accurate, and actionable information to users about various government programs 
        related to areas like education, healthcare, agriculture, and insurance. 
        Your responses should be grounded in the provided context and include details about the scheme name, specific benefits, and eligibility criteria. 
        Ensure the information is delivered in a straightforward, conversational manner without using markdown formatting.
        Example Query: {user_input}
        Context: {context}
        Answer in {user_language}. Language code: {user_language_code}.
        """

        response = llm.invoke(prompt)
        hall_score = hallucination_score(context, response.content)

        return {
            "response": response.content,
            "language": user_language,
            "language_code": user_language_code,
            "confidence_score": hall_score
        }

    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/audio-query")
async def process_audio_query(file: UploadFile = File(...)):
    try:
        # Save the uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
            content = await file.read()
            temp_file.write(content)
            temp_file_path = temp_file.name

        # Process the audio file
        gemini_resp = gemini.speech_to_text(temp_file_path)
        
        # Clean up the temporary file
        os.unlink(temp_file_path)

        # Process the text response
        response_dict = json_chain.invoke({"query": gemini_resp})

        user_language = response_dict['language']
        user_input = response_dict['text']
        user_language_code = response_dict["language_code"]

        docs_with_scores = chroma.similarity_search_with_score(user_input)
        top_docs = docs_with_scores[:4]
        context = " ".join(doc.page_content for doc, _ in top_docs)

        if not context:
            raise HTTPException(status_code=404, detail="No relevant information found")

        prompt = f"""
        You are a highly knowledgeable assistant specializing in Indian government schemes. 
        Your task is to provide clear, accurate, and actionable information to users about various government programs 
        related to areas like education, healthcare, agriculture, and insurance. 
        Your responses should be grounded in the provided context and include details about the scheme name, specific benefits, and eligibility criteria. 
        Ensure the information is delivered in a straightforward, conversational manner without using markdown formatting.
        Example Query: {user_input}
        Context: {context}
        Answer in {user_language}. Language code: {user_language_code}.
        """

        response = llm.invoke(prompt)
        hall_score = hallucination_score(context, response.content)

        return {
            "response": response.content,
            "language": user_language,
            "language_code": user_language_code,
            "confidence_score": hall_score
        }

    except Exception as e:
        logger.error(f"Error processing audio query: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {"status": "healthy"}
