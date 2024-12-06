from os import getenv
from urllib.parse import urljoin

# Configuration parameters by area

# Load text/documents
DATA_PATH = "data"

# Vector Store
VECTOR_DB_PATH = "vector_db"
COLLECTION_NAME = "my_docs"

# Text embedding models choices
EMBEDDING_MODELS = [
    "nomic-ai/nomic-embed-text-v1.5",
    "multi-qa-mpnet-base-cos-v1",
    "all-mpnet-base-v2",
    "multi-qa-MiniLM-L6-cos-v1",
    "all-MiniLM-L6-v2",
]

# LLM (ollama)
LLMBASEURL = urljoin(getenv("OLLAMA_HOST", "http://localhost:11434"), "api")
LLM_CHOICES = [
    "llama3.2",
    "llama3.1",
    "phi3.5",
    "mistral",
    "gemma2",
    "qwen2.5",
    "llama-3-sauerkrautlm",
    "dolphin-llama3",
    "zephyr",
    "qwen2.5:3b",
    "smollm2",
]

# Other Features
EXPORT_PATH = "exports"
CONVERSATION_LENTGH = 10  # Max number of interactions kept for dialogue history context
KEYWORD_SEARCH = True  # Alternative keyword search
FILTER_BY_KEYWORD = True  # Optimize semantic retrieval with keyword

# Internal
LOG_DIR = 'log'
LOG_FILE = 'info.log'
