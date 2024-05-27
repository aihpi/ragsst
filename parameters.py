# General configuration parameters on each area

# Load text/documents
DATA_PATH = "sample_data/"

# Text embedding (sentence transformer)
# EMBEDDING_MODEL = "all-MiniLM-L6-v2"
# EMBEDDING_MODEL = "multi-qa-MiniLM-L6-cos-v1"
EMBEDDING_MODEL = "multi-qa-mpnet-base-cos-v1"

# Vector Store
CHROMA_DATA_PATH = "chroma_data/"
COLLECTION_NAME = "sample_docs"

# LLM (ollama)
LLMBASEURL = "http://localhost:11434/api"
# MODEL = "phi3"
MODEL = "llama3"

# Frontend
GUI_TITLE = f"Local RAG ({MODEL})"
