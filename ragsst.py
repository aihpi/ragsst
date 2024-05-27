import logging
import os
import chromadb
from chromadb.utils import embedding_functions
from tqdm import tqdm
import requests, json
from random import choice
import gradio as gr
from typing import List, Callable, Any
from collections import deque
from utils import list_files, read_file, split_text
from parameters import DATA_PATH, CHROMA_DATA_PATH, EMBEDDING_MODEL, COLLECTION_NAME
from parameters import LLMBASEURL, MODEL


# ============== LLM (Ollama) =================================================


def llm_generate(prompt: str, top_k: int = 5, top_p: float = 0.9, temp: float = 0.2) -> str:
    url = LLMBASEURL + "/generate"
    data = {
        "model": MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": temp, "top_p": top_p, "top_k": top_k},
    }

    try:
        r = requests.post(url, json=data)
        response_dic = json.loads(r.text)
        return response_dic.get('response', '')

    except Exception as e:
        logging.error(e)


def llm_chat(user_message: str, top_k: int = 5, top_p: float = 0.9, temp: float = 0.5) -> str:
    url = LLMBASEURL + "/chat"
    conversation.append({"role": "user", "content": user_message})
    data = {
        "model": MODEL,
        "messages": list(conversation),
        "stream": False,
        "options": {"temperature": temp, "top_p": top_p, "top_k": top_k},
    }

    try:
        r = requests.post(url, json=data)
        response_dic = json.loads(r.text)
        response = response_dic.get('message', '')
        conversation.append(response)
        # print("-" * 100)
        # print("\n".join(map(str, conversation)))
        return response.get('content', '')

    except Exception as e:
        logging.error(e)


# ============== Vector Store =================================================


def make_collection(
    data_path: str, collection_name: str, skip_included_files: bool = True
) -> None:
    """Create vector store collection from a set of documents"""

    vs_client = chromadb.PersistentClient(path=CHROMA_DATA_PATH)
    embedding_func = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBEDDING_MODEL
    )

    print("Populating emeddings database...")
    print(f"Collection: {collection_name}")

    collection = vs_client.get_or_create_collection(
        name=collection_name,
        embedding_function=embedding_func,
        metadata={"hnsw:space": "cosine"},
    )

    files = list_files(data_path, extensions=('.txt', '.pdf'))
    print(f"Found files: {', '.join(files)} ...")

    if skip_included_files:
        sources = {m.get('source') for m in collection.get().get('metadatas')}

    for f in files:
        _, file_name = os.path.split(f)

        if skip_included_files and file_name in sources:
            print(file_name, "Already in Vector-DB, skipping...")
            continue

        print(f"\nReading and splitting {file_name} ...")
        text = read_file(f)
        chunks = split_text(text)
        print("Resulting segments:", len(chunks))

        print(f"\nEmbedding and storing {file_name} ...")

        for i, c in tqdm(enumerate(chunks, 1), total=len(chunks)):
            collection.add(
                documents=c,
                ids=f"id{file_name[:-4]}.{i}",
                metadatas={"source": file_name, "part": i},
            )


# ============== Semantic Search / Retrieval ==================================


def retrieve_content_w_meta_info(
    query: str = '', nresults: int = 2, sim_th: float | None = None
) -> str:
    """Get list of relevant content from a collection including similarity and sources"""

    query_result = collection.query(query_texts=query, n_results=nresults)
    docs_selection = []

    for i in range(len(query_result.get('ids')[0])):

        sim = round(1 - query_result.get('distances')[0][i], 2)

        if sim_th is not None:
            if sim < sim_th:
                continue

        doc = query_result.get('documents')[0][i]
        metadata = str(query_result.get('metadatas')[0][i])
        docs_selection.append('\n'.join([doc, f"Relevance: {sim}", metadata]))

    return "\n-----------------\n\n".join(docs_selection)


def retrieve_content_mockup(query, nresults=2, sim_th=0.25):
    return f"Some Snippet from documents related to {query}\nRelevance:0.7\nSource: holmes.pdf | part: n"


def get_relevant_text(query: str = '', nresults: int = 2, sim_th: float | None = None) -> str:
    """Get relevant text from a collection for a given query"""

    query_result = collection.query(query_texts=query, n_results=nresults)
    docs = query_result.get('documents')[0]
    if sim_th is not None:
        similarities = [1 - d for d in query_result.get("distances")[0]]
        relevant_docs = [d for d, s in zip(docs, similarities) if s >= sim_th]
        return ''.join(relevant_docs)
    return ''.join(docs)


# ============== Retrieval Augemented Generation ==============================


def get_context_prompt(question: str, context: str) -> str:
    contextual_prompt = (
        "Use the following context to answer the question at the end. "
        "Keep the answer as concise as possible.\n"
        "Context:\n"
        f"{context}"
        "\nQuestion:\n"
        f"{question}"
    )

    return contextual_prompt


def rag_query(user_msg: str, top_k: int, top_p: float, temp: float) -> str:
    relevant_text = get_relevant_text(user_msg, sim_th=0.4)
    context_query = get_context_prompt(user_msg, relevant_text)
    bot_response = llm_generate(context_query, top_k=top_k, top_p=top_p, temp=temp)
    return bot_response


def rag_chat(user_msg: str, history: List, top_k: int, top_p: float, temp: float) -> str:
    relevant_text = get_relevant_text(user_msg, sim_th=0.4)
    context_query = get_context_prompt(user_msg, relevant_text)
    bot_response = llm_generate(context_query, top_k=top_k, top_p=top_p, temp=temp)
    return bot_response


# ============== LLM chat w/o Document Context ================================


def chat(user_msg, history, top_k, top_p, temp):
    bot_response = llm_chat(user_msg, top_k=top_k, top_p=top_p, temp=temp)
    return bot_response


# ============== Utils ========================================================


def llm_mockup(prompt, top_k=1, top_p=0.9, temp=0.5):
    return choice(["Yes!", "Not sure", "It depends", "42"])


def chat_mockup(message, history):
    return choice(["Yes!", "Not sure", "It depends", "42"])


def check_initdb_conditions() -> bool:

    return (
        os.path.exists(DATA_PATH)
        and os.listdir(DATA_PATH)
        and (not os.path.exists(CHROMA_DATA_PATH) or not os.listdir(CHROMA_DATA_PATH))
    )


# ============== Interface ====================================================


def make_interface(
    rag_query: Callable,
    semantic_retrieval: Callable,
    rag_chat: Callable,
    chat: Callable,
    makedb: Callable,
) -> Any:

    rag_query_ui = gr.Interface(
        rag_query,
        gr.Textbox(label="Query"),
        gr.Textbox(label="Answer", lines=10),
        description="Query an LLM about information from your documents.",
        allow_flagging="never",
        additional_inputs=[
            gr.Slider(1, 10, value=5, step=1, label="Top k"),
            gr.Slider(0.1, 1, value=0.9, step=0.1, label="Top p"),
            gr.Slider(0.1, 1, value=0.5, step=0.1, label="Temp"),
        ],
        additional_inputs_accordion=gr.Accordion(label="LLM Settings", open=False),
    )

    semantic_retrieval_ui = gr.Interface(
        semantic_retrieval,
        gr.Textbox(label="Query"),
        gr.Textbox(label="Related Content", lines=20),
        description="Find information in your documents.",
        allow_flagging="manual",
        additional_inputs=[
            gr.Slider(1, 5, value=2, step=1, label="Top n Results"),
            gr.Slider(0, 1, value=0.4, step=0.1, label="Relevance threshold"),
        ],
        additional_inputs_accordion=gr.Accordion(label="Retrieval Settings", open=False),
    )

    rag_chat_ui = gr.ChatInterface(
        rag_chat,
        description="Query and interact with an LLM considering your documents information.",
        chatbot=gr.Chatbot(height=700),
        additional_inputs=[
            gr.Slider(1, 10, value=3, step=1, label="Top K"),
            gr.Slider(0.1, 1, value=0.9, step=0.1, label="Top p"),
            gr.Slider(0.1, 1, value=0.3, step=0.1, label="Temp"),
        ],
        additional_inputs_accordion=gr.Accordion(label="LLM Settings", open=False),
    )

    chat_ui = gr.ChatInterface(
        chat,
        description="Simply chat with the LLM, without document context.",
        chatbot=gr.Chatbot(height=700),
        additional_inputs=[
            gr.Slider(1, 10, value=5, step=1, label="Top K"),
            gr.Slider(0.1, 1, value=0.9, step=0.1, label="Top p"),
            gr.Slider(0.1, 1, value=0.5, step=0.1, label="Temp"),
        ],
        additional_inputs_accordion=gr.Accordion(label="LLM Settings", open=False),
    )

    with gr.Blocks() as embed_docs_ui:
        gr.Markdown(
            "Make and populate the Embeddings Database (Vector Store) with your documents."
        )
        data_path = gr.Textbox(value=DATA_PATH, label="Documents Path")
        collection_name = gr.Textbox(value=COLLECTION_NAME, label="Collection Name")
        makedb_btn = gr.Button("Make Db")
        text_output = gr.Textbox(label="Info")
        makedb_btn.click(fn=makedb, inputs=[data_path, collection_name], outputs=text_output)

    gui = gr.TabbedInterface(
        [rag_query_ui, semantic_retrieval_ui, rag_chat_ui, chat_ui, embed_docs_ui],
        ["RAG Query", "Semantic Retrieval", "RAG Chat", "Chat", "Make Db"], title="Local RAGSST",
    )

    return gui


if __name__ == "__main__":

    if check_initdb_conditions():
        make_collection(
            data_path=DATA_PATH,
            collection_name=COLLECTION_NAME,
        )

    MAX_CONVERSATION_LENGTH = 10
    conversation = deque(maxlen=MAX_CONVERSATION_LENGTH)
    vs_client = chromadb.PersistentClient(path=CHROMA_DATA_PATH)
    embedding_func = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBEDDING_MODEL
    )

    print(f"Loading collection {COLLECTION_NAME} ...")
    collection = vs_client.get_collection(name=COLLECTION_NAME, embedding_function=embedding_func)

    mpragst = make_interface(
        rag_query, retrieve_content_w_meta_info, rag_chat, chat, make_collection
    )
    mpragst.launch()
