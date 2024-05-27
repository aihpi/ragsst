import gradio as gr
from random import choice


def rag_bot(message, history):
    return choice(["Yes!", "Not sure", "It depends", "42"])


def chat_bot(message, history):
    "No context, just the LLM"
    return choice(["Yes!", "Not sure", "It depends", "42"])


def semantic_search(query, nresults=2, sim_th=0.4):
    return f"Some Snippet from documents related to {query}\nRelevance:0.7\nSource: holmes.pdf | part: n"


def make_collection(data_path, collection_name):
    return "Making Database..."


rag_query = gr.Interface(
    lambda query: choice(["Yes!", "Not sure", "It depends", "42"]),
    "text",
    "text",
    description="Query an LLM considering information from your docs",
)

semantic_retrieval = gr.Interface(
    semantic_search,
    "text",
    "text",
    description="Find information in your documents",
    additional_inputs=[
        gr.Slider(1, 5, value=2, step=1, label="Top n Results"),
        gr.Slider(0.1, 1, value=0.4, step=0.1, label="Relevance threshold"),
    ],
)

rag_chat = gr.ChatInterface(
    rag_bot, description="Query and interact with an LLM regarding your documents information"
)
chat = gr.ChatInterface(
    chat_bot, description="Simply chat with the LLM. Contextless (not considering any documents)."
)

make_db = gr.Interface(
    fn=make_collection,
    inputs=["text", "text"],
    outputs="text",
    submit_btn="Load",
    clear_btn="Delete DB",
    description="Populate the Vector Store (Database)",
)

ragsss = gr.TabbedInterface(
    [rag_query, semantic_retrieval, rag_chat, chat, make_db],
    ["RAG Query", "Semantic Retrieval", "RAG Chat", "Chat", "Make DB"],
)

if __name__ == "__main__":
    ragsss.launch()
