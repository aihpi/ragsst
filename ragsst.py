import logging
import os
import chromadb
from chromadb.utils import embedding_functions
from tqdm import tqdm
import requests, json
from random import choice
import gradio as gr
from typing import List, Any
from collections import deque
from utils import list_files, read_file, split_text
from parameters import DATA_PATH, VECTOR_DB_PATH, EMBEDDING_MODEL, COLLECTION_NAME
from parameters import LLMBASEURL, MODEL

logging.basicConfig(format=os.getenv('LOG_FORMAT', '%(asctime)s [%(levelname)s] %(message)s'))
logger = logging.getLogger(__name__)
logger.setLevel(os.getenv('LOG_LEVEL', logging.INFO))


class RAGTools:
    def __init__(
        self,
        model: str = MODEL,
        llm_base_url: str = LLMBASEURL,
        data_path: str = DATA_PATH,
        embedding_model: str = EMBEDDING_MODEL,
        collection_name: str = COLLECTION_NAME,
    ):
        self.model = model
        self.llm_base_url = llm_base_url
        self.max_conversation_length = 10
        self.conversation = deque(maxlen=self.max_conversation_length)
        self.rag_conversation = deque(maxlen=self.max_conversation_length)
        self.data_path = data_path
        self.embedding_model = embedding_model
        self.vs_client = chromadb.PersistentClient(path=VECTOR_DB_PATH)
        self.embedding_func = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=self.embedding_model
        )
        self.collection_name = collection_name
        if self.check_initdb_conditions():
            logger.debug("Init DB contitions are met")
            self.make_collection(data_path=self.data_path, collection_name=self.collection_name)
        else:
            self.collection = self.vs_client.get_collection(
                name=self.collection_name, embedding_function=self.embedding_func
            )

    # ============== LLM (Ollama) ==============================================

    def llm_generate(
        self, prompt: str, top_k: int = 5, top_p: float = 0.9, temp: float = 0.2
    ) -> str:
        url = self.llm_base_url + "/generate"
        data = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": temp, "top_p": top_p, "top_k": top_k},
        }

        try:
            r = requests.post(url, json=data)
            response_dic = json.loads(r.text)
            return response_dic.get('response', '')

        except Exception as e:
            logger.error(f"Exception: {e}\nResponse:{response_dic}")

    def llm_chat(
        self, user_message: str, top_k: int = 5, top_p: float = 0.9, temp: float = 0.5
    ) -> str:

        url = self.llm_base_url + "/chat"
        self.conversation.append({"role": "user", "content": user_message})
        data = {
            "model": self.model,
            "messages": list(self.conversation),
            "stream": False,
            "options": {"temperature": temp, "top_p": top_p, "top_k": top_k},
        }

        try:
            r = requests.post(url, json=data)
            response_dic = json.loads(r.text)
            response = response_dic.get('message', '')
            self.conversation.append(response)
            logger.debug("-" * 100)
            logger.debug("\n".join(map(str, self.conversation)))
            return response.get('content', '')

        except Exception as e:
            logger.error(f"Exception: {e}\nResponse:{response_dic}")

    def list_local_models(self) -> List:

        url = self.llm_base_url + "/tags"

        try:
            r = requests.get(url)
            response_dic = json.loads(r.text)
            models_names = [model.get("name") for model in response_dic.get("models")]
            return models_names

        except Exception as e:
            logger.error(f"Exception: {e}\nResponse:{response_dic}")

    # ============== Vector Store ==============================================

    def make_collection(
        self,
        data_path: str,
        collection_name: str,
        skip_included_files: bool = True,
    ) -> None:
        """Create vector store collection from a set of documents"""

        logger.info(
            f"Documents Path: {data_path} | Collection Name: {collection_name} | Embedding Model: {self.embedding_model}"
        )

        self.embedding_func = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=self.embedding_model
        )
        self.collection = self.vs_client.get_or_create_collection(
            name=collection_name,
            embedding_function=self.embedding_func,
            metadata={"hnsw:space": "cosine"},
        )

        files = list_files(data_path, extensions=('.txt', '.pdf'))
        logger.info(f"{len(files)} files found.")
        logger.info(f"Files: {', '.join([f.replace(data_path, '') for f  in files])}")
        logger.info("Populating embeddings database...")

        if skip_included_files:
            sources = {m.get('source') for m in self.collection.get().get('metadatas')}

        for f in files:
            _, file_name = os.path.split(f)

            if skip_included_files and file_name in sources:
                logger.info(f"{file_name} Already in Vector-DB, skipping...")
                continue

            logger.info(f"Reading and splitting {file_name} ...")
            text = read_file(f)
            chunks = split_text(text)
            logger.info(f"Resulting segment count: {len(chunks)}")
            logger.info(f"Embedding and storing {file_name} ...")

            for i, c in tqdm(enumerate(chunks, 1), total=len(chunks)):
                self.collection.add(
                    documents=c,
                    ids=f"id{file_name[:-4]}.{i}",
                    metadatas={"source": file_name, "part": i},
                )

        logger.debug(f"Stored Collections: {self.vs_client.list_collections()}")

    # ============== Semantic Search / Retrieval ===============================

    def retrieve_content_mockup(self, query, nresults=2, sim_th=0.25):
        return f"Some Snippet from documents related to {query}\nRelevance:0.7\nSource: holmes.pdf | part: n"

    def retrieve_content_w_meta_info(
        self, query: str = '', nresults: int = 2, sim_th: float | None = None
    ) -> str:
        """Get list of relevant content from a collection including similarity and sources"""

        query_result = self.collection.query(query_texts=query, n_results=nresults)

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

    def get_relevant_text(
        self, query: str = '', nresults: int = 2, sim_th: float | None = None
    ) -> str:
        """Get relevant text from a collection for a given query"""

        query_result = self.collection.query(query_texts=query, n_results=nresults)
        docs = query_result.get('documents')[0]
        if sim_th is not None:
            similarities = [1 - d for d in query_result.get("distances")[0]]
            relevant_docs = [d for d, s in zip(docs, similarities) if s >= sim_th]
            return ''.join(relevant_docs)
        return ''.join(docs)

    # ============== Retrieval Augemented Generation ===========================

    def get_context_prompt(self, query: str, context: str) -> str:
        contextual_prompt = (
            "Use the following context to answer the query at the end. "
            "Keep the answer as concise as possible.\n"
            "Context:\n"
            f"{context}"
            "\nQuery:\n"
            f"{query}"
        )

        return contextual_prompt

    def get_condenser_prompt(self, query: str, chat_history: List) -> str:
        history = '\n'.join(list(chat_history))
        condenser_prompt = (
            "Given the following chat history and a follow up query, rephrase the follow up query to be a standalone query. "
            "Just create the standalone query without commentary. Use the same language."
            "\nChat history:\n"
            f"{history}"
            f"\nFollow Up Query: {query}"
            "\nStandalone Query:"
        )
        return condenser_prompt

    def rag_query(
        self, user_msg: str, sim_th: float, nresults: int, top_k: int, top_p: float, temp: float
    ) -> str:
        logger.debug(
            f"rag_query args: sim_th: {sim_th}, nresults: {nresults}, top_k: {top_k}, top_p: {top_p}, temp: {temp}"
        )
        relevant_text = self.get_relevant_text(user_msg, nresults=nresults, sim_th=sim_th)
        # logger.debug(f"\nRelevant Context:\n{relevant_text}")
        if not relevant_text:
            return "Relevant passage not found. Try lowering the relevance threshold."
        contextualized_query = self.get_context_prompt(user_msg, relevant_text)
        bot_response = self.llm_generate(contextualized_query, top_k=top_k, top_p=top_p, temp=temp)
        return bot_response

    def rag_chat(
        self,
        user_msg: str,
        ui_hist: List,
        sim_th: float,
        nresults: int,
        top_k: int,
        top_p: float,
        temp: float,
    ) -> str:
        logger.debug(
            f"rag_chat args: sim_th: {sim_th}, nresults: {nresults}, top_k: {top_k}, top_p: {top_p}, temp: {temp}"
        )
        MSG_NO_CONTEXT = "Relevant passage not found."

        if not self.rag_conversation:
            relevant_text = self.get_relevant_text(user_msg, nresults=nresults, sim_th=sim_th)
            if not relevant_text:
                return MSG_NO_CONTEXT
            self.rag_conversation.append('Query: ' + user_msg)
            contextualized_query = self.get_context_prompt(user_msg, relevant_text)
            bot_response = self.llm_generate(
                contextualized_query, top_k=top_k, top_p=top_p, temp=temp
            )
            self.rag_conversation.append('Answer: ' + bot_response)
            return bot_response

        condenser_prompt = self.get_condenser_prompt(user_msg, self.rag_conversation)
        logger.debug(f"\nCondenser prompt:\n{condenser_prompt}")

        standalone_query = self.llm_generate(condenser_prompt, top_k=top_k, top_p=top_p, temp=temp)
        logger.debug(f"Standalone query: {standalone_query}")

        relevant_text = self.get_relevant_text(standalone_query, nresults=nresults, sim_th=sim_th)
        if not relevant_text:
            return MSG_NO_CONTEXT
        contextualized_standalone_query = self.get_context_prompt(standalone_query, relevant_text)
        # logger.debug(f"Contextualized new query:\n{contextualized_standalone_query}")

        bot_response = self.llm_generate(
            contextualized_standalone_query, top_k=top_k, top_p=top_p, temp=temp
        )
        self.rag_conversation.append('Query:\n' + standalone_query)
        self.rag_conversation.append('Answer:\n' + bot_response)
        return bot_response

    # ============== LLM chat w/o Document Context =============================

    def chat(self, user_msg, history, top_k, top_p, temp):
        bot_response = self.llm_chat(user_msg, top_k=top_k, top_p=top_p, temp=temp)
        return bot_response

    # ============== Utils =====================================================

    def llm_mockup(self, prompt, top_k=1, top_p=0.9, temp=0.5):
        return choice(["Yes!", "Not sure", "It depends", "42"])

    def chat_mockup(self, message, history):
        return choice(["Yes!", "Not sure", "It depends", "42"])

    def check_initdb_conditions(self) -> bool:

        return (
            os.path.exists(self.data_path)
            and os.listdir(self.data_path)
            and (
                not os.path.exists(VECTOR_DB_PATH)
                or not [f.path for f in os.scandir(VECTOR_DB_PATH) if f.is_dir()]
            )
        )

    def set_model(self, llm: str):
        self.model = llm
        logger.info(f"Chosen Model: {self.model}")

    def set_embeddings_model(self, emb_model: str):
        self.embedding_model = emb_model
        self.embedding_func = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=self.embedding_model
        )
        logger.debug(f"Embedding Model: {self.embedding_model}")

    def set_data_path(self, data_path: str):
        self.data_path = data_path
        logger.debug(f"Data Path: {self.data_path}")

    def set_collection_name(self, collection_name: str):
        self.collection_name = collection_name
        logger.debug(f"Collection Name: {self.collection_name}")


# ============== Interface ====================================================


def make_interface(ragsst: RAGTools) -> Any:

    # Parameter information
    pinfo = {
        "Rth": "Set the relevance level for the content retrieval",
        "TopnR": "Select the maximum number of passages to retrieve",
        "Top k": "LLM Parameter. A higher value will produce more varied text",
        "Top p": "LLM Parameter. A higher value will produce more varied text",
        "Temp": "LLM Parameter. Higher values increase the randomness of the answer",
    }

    rag_query_ui = gr.Interface(
        ragsst.rag_query,
        gr.Textbox(label="Query"),
        gr.Textbox(label="Answer", lines=14),
        description="Query an LLM about information from your documents.",
        allow_flagging="never",
        additional_inputs=[
            gr.Slider(
                0, 1, value=0.3, step=0.1, label="Relevance threshold", info=pinfo.get("Rth")
            ),
            gr.Slider(1, 5, value=3, step=1, label="Top n results", info=pinfo.get("TopnR")),
            gr.Slider(1, 10, value=5, step=1, label="Top k", info=pinfo.get("Top k")),
            gr.Slider(
                0.1, 1, value=0.9, step=0.1, label="Top p", info=pinfo.get("Top p"), visible=False
            ),
            gr.Slider(0.1, 1, value=0.3, step=0.1, label="Temp", info=pinfo.get("Temp")),
        ],
        additional_inputs_accordion=gr.Accordion(label="Settings", open=False),
    )

    semantic_retrieval_ui = gr.Interface(
        ragsst.retrieve_content_w_meta_info,
        gr.Textbox(label="Query"),
        gr.Textbox(label="Related Content", lines=20),
        description="Find information in your documents.",
        allow_flagging="manual",
        additional_inputs=[
            gr.Slider(1, 5, value=2, step=1, label="Top n results", info=pinfo.get("TopnR")),
            gr.Slider(
                0, 1, value=0.4, step=0.1, label="Relevance threshold", info=pinfo.get("Rth")
            ),
        ],
        additional_inputs_accordion=gr.Accordion(label="Retrieval Settings", open=False),
    )

    rag_chat_ui = gr.ChatInterface(
        ragsst.rag_chat,
        description="Query and interact with an LLM considering your documents information.",
        chatbot=gr.Chatbot(height=500),
        additional_inputs=[
            gr.Slider(
                0, 1, value=0.4, step=0.1, label="Relevance threshold", info=pinfo.get("Rth")
            ),
            gr.Slider(1, 5, value=3, step=1, label="Top n results", info=pinfo.get("TopnR")),
            gr.Slider(1, 10, value=3, step=1, label="Top k", info=pinfo.get("Top k")),
            gr.Slider(
                0.1, 1, value=0.9, step=0.1, label="Top p", info=pinfo.get("Top p"), visible=False
            ),
            gr.Slider(0.1, 1, value=0.3, step=0.1, label="Temp", info=pinfo.get("Temp")),
        ],
        additional_inputs_accordion=gr.Accordion(label="Settings", open=False),
    )

    chat_ui = gr.ChatInterface(
        ragsst.chat,
        description="Simply chat with the LLM, without document context.",
        chatbot=gr.Chatbot(height=500),
        additional_inputs=[
            gr.Slider(1, 10, value=5, step=1, label="Top k", info=pinfo.get("Top k")),
            gr.Slider(0.1, 1, value=0.9, step=0.1, label="Top p", info=pinfo.get("Top p")),
            gr.Slider(0.1, 1, value=0.5, step=0.1, label="Temp", info=pinfo.get("Temp")),
        ],
        additional_inputs_accordion=gr.Accordion(label="LLM Settings", open=False),
    )

    with gr.Blocks() as config_ui:
        with gr.Row():
            with gr.Column(scale=3):

                def make_db(data_path, collection_name, embedding_model):
                    ragsst.set_data_path(data_path)
                    ragsst.set_collection_name(collection_name)
                    ragsst.set_embeddings_model(embedding_model)
                    ragsst.make_collection(data_path, collection_name)

                gr.Markdown(
                    "Make and populate the Embeddings Database (Vector Store) with your documents."
                )
                with gr.Row():
                    data_path = gr.Textbox(value=DATA_PATH, label="Documents Path")
                    collection_name = gr.Textbox(value=COLLECTION_NAME, label="Collection Name")
                emb_model = gr.Dropdown(
                    choices=[EMBEDDING_MODEL, "all-MiniLM-L6-v2", "multi-qa-MiniLM-L6-cos-v1"],
                    value=EMBEDDING_MODEL,
                    label="Embedding Model",
                    interactive=True,
                )
                makedb_btn = gr.Button("Make Db", size='sm')
                text_output = gr.Textbox(label="Info")
                makedb_btn.click(
                    fn=make_db,
                    inputs=[data_path, collection_name, emb_model],
                    outputs=text_output,
                )

            with gr.Column(scale=2):
                gr.Markdown("Choose the Language Model")
                model_choices = ragsst.list_local_models()
                print(model_choices)
                model_name = gr.Dropdown(
                    choices=model_choices,
                    allow_custom_value=True,
                    value=MODEL,
                    label="LLM",
                    interactive=True,
                )
                setllm_btn = gr.Button("Submit Choice", size='sm')
                setllm_btn.click(fn=ragsst.set_model, inputs=model_name)

    gui = gr.TabbedInterface(
        [rag_query_ui, semantic_retrieval_ui, rag_chat_ui, chat_ui, config_ui],
        ["RAG Query", "Semantic Retrieval", "RAG Chat", "Chat", "Rag Tool Settings"],
        title="Local RAG Tool",
    )

    return gui


if __name__ == "__main__":

    ragsst = RAGTools()

    mpragst = make_interface(ragsst)
    mpragst.launch()
