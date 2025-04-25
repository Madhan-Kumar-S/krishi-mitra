from dotenv import load_dotenv
import os
import getpass as getpass
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings

load_dotenv()

# Set up Google Cloud credentials immediately
#credentials_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "krishi-mitra-test-43a53d2071ac.json")
#if not os.path.exists(credentials_path):
 #   raise FileNotFoundError(f"Credentials file not found at: {credentials_path}")

#os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = credentials_path
#print("GOOGLE_APPLICATION_CREDENTIALS set to:", credentials_path)

os.environ["GOOGLE_API_KEY"]="AIzaSyAXoIWl2lswnJxpE3srSx1DJak1uVNc0Ds"

# Set USER_AGENT to identify requests
os.environ["USER_AGENT"] = "KrishiMitra/1.0"

# Verify the credentials file is readable
# try:
#     with open(credentials_path, 'r') as f:
#         import json
#         json.load(f)
#     print("Credentials file is valid JSON")
# except Exception as e:
#     print(f"Error reading credentials file: {e}")
#     raise

CHUNK_SIZE = 2400
CHUNK_OVERLAP = 200

BASE_URL = "https://api.myscheme.gov.in/search/v4/schemes?lang=en&q=%5B%5D&keyword=&sort=&size="
BASE_SCHEME_URL = "https://www.myscheme.gov.in/schemes/"
HEADERS = {
    'accept': 'application/json, text/plain, */*',
    'accept-language': 'en-US,en;q=0.9',
    'origin': 'https://www.myscheme.gov.in',
    'priority': 'u=1, i',
    'sec-ch-ua': '"Not/A)Brand";v="8", "Chromium";v="126", "Google Chrome";v="126"',
    'sec-ch-ua-mobile': '?0',
    'sec-ch-ua-platform': '"macOS"',
    'sec-fetch-dest': 'empty',
    'sec-fetch-mode': 'cors',
    'sec-fetch-site': 'same-site',
    'user-agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36',
    'x-api-key': '<add your X-API-KEY for this website'
}
TOTAL_RESULTS = 2389
MAX_SIZE = 100
# EMBEDDINGS = HuggingFaceEmbeddings(
#     model_name="sentence-transformers/all-mpnet-base-v2",
#     model_kwargs={"device": "cuda"},
# )

# Initialize embeddings with explicit credentials
EMBEDDINGS = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001",
    google_api_key="AIzaSyAXoIWl2lswnJxpE3srSx1DJak1uVNc0Ds",  # We're using service account credentials instead
    #credentials_path=credentials_path
)
START_WEB_SCRAPING_MYSCHEMES = False

# This function is kept for backward compatibility
def set_envs():
    # The credentials are already set at module import
    pass
