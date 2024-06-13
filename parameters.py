# Configuration parameters by area

# Load text/documents
DATA_PATH = "sample_data"

# Text embedding (sentence transformer)
EMBEDDING_MODEL = "multi-qa-mpnet-base-cos-v1"

# Vector Store
VECTOR_DB_PATH = "chroma_data"
COLLECTION_NAME = "sample_docs"

# LLM (ollama)
LLMBASEURL = "http://localhost:11434/api"
MODEL = "llama3"

# Frontend
GUI_TITLE = f"Local RAG ({MODEL})"
