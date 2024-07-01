from os import getenv

# Configuration parameters by area

# Load text/documents
DATA_PATH = "data"

# Vector Store
VECTOR_DB_PATH = "vector_db"
COLLECTION_NAME = "my_docs"

# Text embedding models choices
EMBEDDING_MODELS = [
    "multi-qa-mpnet-base-cos-v1",
    "all-mpnet-base-v2",
    "multi-qa-MiniLM-L6-cos-v1",
    "all-MiniLM-L6-v2",
]

# LLM (ollama)
LLMBASEURL = getenv("OLLAMA_HOST", "http://localhost:11434/api")
LLM_CHOICES = ["llama3", "phi3", "mistral", "gemma", "qwen2", "dolphin-llama3", "zephyr"]

# Other Features
EXPORT_PATH = "exports"
CONVERSATION_LENTGH = 10  # Max number of interactions kept for dialogue history context
