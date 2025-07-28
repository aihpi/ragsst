# %% [markdown]

# # Workshop Outline
# # Local RAG System
#
# <img src="images/ragflowchart.png" width="700" />
#
# ## • Embedding of text

# - Read/Load Document (txt, pdf, etc.)
# - Split text into chunks considering embedding limit and basic context
# - Encode text into vector
# - Find Similarity

# ## &bull; Vector store

# - Using a Vector Database
# - Make a collection
# - Query collection. Find related text.

# ## &bull; Query LLM with contextual data

# -----------------------------------------------------------------------------

# ## Hands on...

# #### &bull; **Embedding of text**

# %%
# Load Embedding model
# https://www.sbert.net/
# pip3 install sentence-transformers

from sentence_transformers import SentenceTransformer

EMBEDDING_MODEL = "all-MiniLM-L6-v2"
model = SentenceTransformer(EMBEDDING_MODEL)

# %%
# Sample Text

texts = [
    "Bali's beautiful beaches and rich culture stands out as a fantastic travel destination.",
    "Pizza in Rome is famous for its thin crust, fresh ingredients and wood-fired ovens.",
    "Graphics processing units (GPU) have become an essential foundation for artificial intelligence.",
    "Newton's laws of motion transformed our understanding of physics.",
    "The French Revolution played a crucial role in shaping contemporary France.",
    "Maintaining good health requires regular exercise, balanced diet and quality sleep.",
    "Dali's surrealistic artworks, like 'The Persistence of Memory,' captivate audiences with their dreamlike imagery and imaginative brilliance",
    "Global warming threatens the planet's ecosystems and wildlife.",
    "The KI-Servicezentrum Berlin-Brandenburg offers services such as consulting, workshops, MOOCs and computer resources.",
    "Django Reinhardt's jazz compositions are celebrated for their captivating melodies and innovative guitar work..",
]

# %%
# Encode texts

text_embeddings = model.encode(texts)

print("Embeddings type:", type(text_embeddings))
print("Embeddings Matrix shape:", text_embeddings.shape)

# Note: "all-MiniLM-L6-v2" encodes texts up to 256 words. It’ll truncate any text longer than this.
# https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2

# %% [markdown]
# #### &bull; **Check Similarity**
# %%
# Storing Embeddings in a simple dict

text_embs_dict = dict(zip(texts, list(text_embeddings)))
# key: text, value: numpy array with embedding

# %%
# Define similarity metric

from numpy import dot
from numpy.linalg import norm


def cos_sim(x, y):
    return dot(x, y) / (norm(x) * norm(y))


# %%
# Check similarities

test_text = "I really need some vacations"
emb_test_text = model.encode(test_text)

print(f"\nCosine similarities for: '{test_text}'\n")
for k, v in text_embs_dict.items():
    print(k, round(cos_sim(emb_test_text, v), 3))

# %% [markdown]
# **To do:**
#
# - Experiment with different sentences and compare similarities.
#
# - Check alternative sentence-transformers. Read Model Cards. Compare results.
# %% [markdown]
# #### &bull; **Using a Vector Database**

# %%

import chromadb
from chromadb.utils import embedding_functions

client = chromadb.Client()
# client = chromadb.PersistentClient(path="chroma_data/")
# For a ChromaDB instance on the disk

embedding_func = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name=EMBEDDING_MODEL
)

# %%
# Init collection

COLLECTION_NAME = "demo_docs"

collection = client.create_collection(
    name=COLLECTION_NAME,
    embedding_function=embedding_func,
    metadata={"hnsw:space": "cosine"},
)

# %%
# Create collection
# (Adding metadata. Optional)

topics = [
    "travel",
    "food",
    "technology",
    "science",
    "history",
    "health",
    "painting",
    "climate change",
    "business",
    "music",
]

collection.add(
    documents=texts,
    ids=[f"id{i}" for i in range(len(texts))],
    metadatas=[{"topic": topic} for topic in topics],
)

# %%
# Quering the Database

query1 = "I am looking for something to eat"
query2 = "What's going on in the world?"
query3 = "Great jazz player"

queries = [query1, query2, query3]

query_results = collection.query(
    query_texts=queries,  # list of strings or just one element (string)
    n_results=2,
)
print("Query dict keys:")
print(query_results.keys())

for i in range(len(queries)):
    print("\nQuery:", queries[i])

    print("\nResults:")
    for j in range(len(query_results["ids"][i])):
        print("id:", query_results["ids"][i][j])
        print("Text:", query_results["documents"][i][j])
        print("Distance:", round(query_results["distances"][i][j], 2))
        print("Metadata:", query_results["metadatas"][i][j])

    print(80 * "-")

# %% [markdown]
# #### &bull; **Running a Large Language Model locally with Ollama**
# %%

import requests, json
from os import getenv
from urllib.parse import urljoin

# $ ollama serve
BASEURL = urljoin(getenv("OLLAMA_HOST", "http://localhost:11434"), "api")
MODEL = "llama3.2"


def generate(prompt, context=[], top_k=5, top_p=0.9, temp=0.5):
    url = BASEURL + "/generate"
    data = {
        "prompt": prompt,
        "model": MODEL,
        "stream": False,
        "context": context,
        "options": {"temperature": temp, "top_p": top_p, "top_k": top_k},
    }

    try:
        r = requests.post(url, json=data)
        response_dic = json.loads(r.text)
        return response_dic.get('response', ''), response_dic.get('context', '')

    except Exception as e:
        print(e)


# %%

llm_response, _ = generate("Hi, who are you", top_k=10, top_p=0.9, temp=0.5)
print(llm_response)

# %% [markdown]
# #### &bull; **Make a simple local LLM chatbot**
# %%
user_input = "Hi. who are you?"
ollama_context = []
print(f"Start chatting with {MODEL} model (Press q to quit)\n")
while user_input != "q":
    bot_response, ollama_context = generate(
        user_input, context=ollama_context, top_k=10, top_p=0.9, temp=0.5
    )
    print("Model message:")
    print(bot_response)
    user_input = input("\nYour prompt: ")

# %% [markdown]
# **To do:**
#
# - Experiment with different input parameters, arguments, prompts and compare results.
# - Try different models (e.g. mistral, llama3.1, gemma2, qwen2.5, llama3.2, qwen2.5:3b, etc.)
#
# %% [markdown]
# -----------------------------------------------------------------------------
# ## Integration
# #### **Integrate the components for a RAG System.**
# ### References:
# #### &bull; **Text embedding: Sentence Bert**
# https://www.sbert.net/
#
# Sample Sentence Transformers:
#
# https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2
#
# https://huggingface.co/sentence-transformers/multi-qa-MiniLM-L6-cos-v1
#
# https://huggingface.co/sentence-transformers/multi-qa-mpnet-base-cos-v1
#
# Check and compare
# #### &bull; **Vector Database: ChromaDB**
# https://docs.trychroma.com/
# #### &bull; **Local LLM: ollama**
# https://ollama.ai/
# #### &bull; **Frontend: Gradio**
# https://www.gradio.app/
