# Loads environment variables and defines all constants used
# across the application. Import from here instead of
# hard-coding values in individual modules.

import os
from dotenv import load_dotenv

# Hard-disable ChromaDB telemetry to stop posthog client crashes
os.environ["ANONYMIZED_TELEMETRY"] = "False"
os.environ["CHROMA_TELEMETRY_IMPL"] = "None"

# Load environment variables from .env file
load_dotenv()

# ---- Groq (Primary LLM — fast cloud inference) ----
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = "llama-3.3-70b-versatile"
GROQ_VISION_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"  # For image understanding
GROQ_BASE_URL = "https://api.groq.com/openai/v1/chat/completions"

# ---- Google Gemini (Secondary LLM) ----
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL = "gemini-2.0-flash"

# ---- Ollama (Final Fallback LLM — local/offline) ----
OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_MODEL = "gpt-oss:20b"

# ---- Embedding Model (Local) ----
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# ---- ChromaDB (Vector Store) ----
CHROMA_PERSIST_DIR = "./chroma_db"

# Collection names for different data types
COLLECTION_DOCUMENTS = "documents"
COLLECTION_EXCEL = "excel"
COLLECTION_IMAGES = "images"

# ---- Text Chunking Parameters ----
CHUNK_SIZE = 500       # Maximum characters per chunk
CHUNK_OVERLAP = 50     # Overlap between consecutive chunks

# ---- Retrieval Settings ----
TOP_K_RESULTS = 4      # Number of top results to return per query

# ---- Tesseract OCR Path (Windows) ----
TESSERACT_PATH = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# ---- Conversation Memory ----
MEMORY_MAX_MESSAGES = 10  # Keep the last N messages in memory
