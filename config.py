import os
from dotenv import load_dotenv

# Hard-disable ChromaDB telemetry
os.environ["ANONYMIZED_TELEMETRY"] = "False"
os.environ["CHROMA_TELEMETRY_IMPL"] = "None"

# Load .env locally; on Streamlit Cloud, inject secrets into env
load_dotenv()

# ---- Groq ----
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = "llama-3.3-70b-versatile"
GROQ_VISION_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"
GROQ_BASE_URL = "https://api.groq.com/openai/v1/chat/completions"

# ---- Gemini ----
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL = "gemini-2.0-flash"

# ---- Ollama ----
OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_MODEL = "gpt-oss:20b"

# ---- Embedding Model ----
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# ---- ChromaDB ----
CHROMA_PERSIST_DIR = "./chroma_db"
COLLECTION_DOCUMENTS = "documents"
COLLECTION_EXCEL = "excel"
COLLECTION_IMAGES = "images"

# ---- Text Chunking ----
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

# ---- Retrieval ----
TOP_K_RESULTS = 4

# ---- Tesseract OCR Path (Windows only, ignored on cloud) ----
TESSERACT_PATH = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# ---- Memory ----
MEMORY_MAX_MESSAGES = 10