from os import getenv

# Configuration parameters by area

# Load text/documents
DATA_PATH = "data"

# Text embedding (sentence transformer)
EMBEDDING_MODEL = "multi-qa-mpnet-base-cos-v1"

# Vector Store
VECTOR_DB_PATH = "vector_db"
COLLECTION_NAME = "my_docs"

# LLM (ollama)
LLMBASEURL = getenv("OLLAMA_HOST", "http://localhost:11434/api")
MODEL = "llama3"
