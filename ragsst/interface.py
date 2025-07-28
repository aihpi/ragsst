import os
import gradio as gr
from ragsst.ragtool import RAGTool
from typing import Any
import ragsst.parameters as p

MODEL = p.LLM_CHOICES[0]
EMBEDDING_MODEL = p.EMBEDDING_MODELS[0]


def make_interface(ragsst: RAGTool) -> Any:

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
        allow_flagging="manual",
        flagging_dir=os.path.join(p.EXPORT_PATH, "rag_query"),
        flagging_options=[("Export", "export")],
        additional_inputs=[
            gr.Slider(
                0, 1, value=0.5, step=0.1, label="Relevance threshold", info=pinfo.get("Rth")
            ),
            gr.Slider(1, 5, value=3, step=1, label="Top n results", info=pinfo.get("TopnR")),
            gr.Slider(1, 10, value=5, step=1, label="Top k", info=pinfo.get("Top k")),
            gr.Slider(
                0.1, 1, value=0.9, step=0.1, label="Top p", info=pinfo.get("Top p"), visible=False
            ),
            gr.Slider(0.1, 1, value=0.3, step=0.1, label="Temp", info=pinfo.get("Temp")),
        ],
        additional_inputs_accordion=gr.Accordion(label="Settings", open=False),
        clear_btn=None,
    )

    semantic_retrieval_ui = gr.Interface(
        ragsst.retrieve_content_w_meta_info,
        gr.Textbox(label="Query"),
        gr.Textbox(label="Related Content", lines=20),
        description="Find information in your documents.",
        allow_flagging="manual",
        flagging_dir=os.path.join(p.EXPORT_PATH, "semantic_retrieval"),
        flagging_options=[("Export", "export")],
        additional_inputs=[
            gr.Slider(1, 5, value=3, step=1, label="Top n results", info=pinfo.get("TopnR")),
            gr.Slider(
                0, 1, value=0.5, step=0.1, label="Relevance threshold", info=pinfo.get("Rth")
            ),
        ],
        additional_inputs_accordion=gr.Accordion(label="Retrieval Settings", open=False),
        clear_btn=None,
    )

    with gr.ChatInterface(
        ragsst.rag_chat,
        description="Query and interact with an LLM considering your documents information.",
        chatbot=gr.Chatbot(height=500),
        additional_inputs=[
            gr.Slider(
                0, 1, value=0.5, step=0.1, label="Relevance threshold", info=pinfo.get("Rth")
            ),
            gr.Slider(1, 5, value=3, step=1, label="Top n results", info=pinfo.get("TopnR")),
            gr.Slider(1, 10, value=5, step=1, label="Top k", info=pinfo.get("Top k")),
            gr.Slider(
                0.1, 1, value=0.9, step=0.1, label="Top p", info=pinfo.get("Top p"), visible=False
            ),
            gr.Slider(0.1, 1, value=0.3, step=0.1, label="Temp", info=pinfo.get("Temp")),
        ],
        additional_inputs_accordion=gr.Accordion(label="Settings", open=False),
        undo_btn=None,
    ) as rag_chat_ui:
        rag_chat_ui.clear_btn.click(ragsst.clear_ragchat_hist)

    with gr.ChatInterface(
        ragsst.chat,
        description="Simply chat with the LLM, without document context.",
        chatbot=gr.Chatbot(height=500),
        additional_inputs=[
            gr.Slider(1, 10, value=5, step=1, label="Top k", info=pinfo.get("Top k")),
            gr.Slider(0.1, 1, value=0.9, step=0.1, label="Top p", info=pinfo.get("Top p")),
            gr.Slider(0.1, 1, value=0.5, step=0.1, label="Temp", info=pinfo.get("Temp")),
        ],
        additional_inputs_accordion=gr.Accordion(label="LLM Settings", open=False),
        undo_btn=None,
    ) as chat_ui:
        chat_ui.clear_btn.click(ragsst.clear_chat_hist)

    with gr.Blocks() as config_ui:

        def read_logs():
            with open(os.path.join(p.LOG_DIR, p.LOG_FILE), "r") as f:
                return f.read()

        with gr.Row():
            with gr.Column(scale=3):

                def make_db(data_path, collection_name, embedding_model):
                    if collection_name is None:
                        collection_name = p.COLLECTION_NAME
                    ragsst.set_data_path(data_path)
                    ragsst.set_embeddings_model(embedding_model)
                    ragsst.make_collection(data_path, collection_name)

                gr.Markdown("Make and populate the Embeddings Database.")
                with gr.Row():
                    with gr.Column():
                        data_path = gr.Textbox(
                            value=ragsst.data_path,
                            label="Documents Path",
                            info="Folder containing your documents",
                            interactive=True,
                        )
                    with gr.Column():
                        collection_choices = ragsst.list_collections_names()
                        collection_name = gr.Dropdown(
                            info="Choose a collection to use/delete or write a name (no spaces allowed) to create a new one",
                            choices=collection_choices,
                            allow_custom_value=True,
                            value=ragsst.collection_name,
                            label="Collection Name",
                            interactive=True,
                        )
                        with gr.Row():
                            setcollection_btn = gr.Button("Set Choice", size='sm')
                            deletecollection_btn = gr.Button("Delete", size='sm')

                        def update_collections_list(current_value):
                            local_collections = ragsst.list_collections_names()
                            if local_collections:
                                if current_value in local_collections:
                                    default_value = current_value
                                else:
                                    default_value = local_collections[0]
                            else:
                                default_value = None
                            return gr.Dropdown(
                                choices=local_collections,
                                value=default_value,
                                interactive=True,
                            )

                emb_model = gr.Dropdown(
                    choices=p.EMBEDDING_MODELS,
                    value=EMBEDDING_MODEL,
                    label="Embedding Model",
                    interactive=True,
                )

                setcollection_btn.click(ragsst.set_collection, inputs=[collection_name, emb_model])
                deletecollection_btn.click(ragsst.delete_collection, inputs=collection_name)

                with gr.Row():
                    makedb_btn = gr.Button("Make/Update Database", size='lg', scale=2)
                    deletedb_btn = gr.Button("Clean Database", size='lg', scale=1)
                info_output = gr.Textbox(read_logs, label="Info", lines=10, every=2)
                makedb_btn.click(
                    fn=make_db,
                    inputs=[data_path, collection_name, emb_model],
                    outputs=info_output,
                )
                deletedb_btn.click(fn=ragsst.clean_database)
                info_output.change(update_collections_list, collection_name, collection_name)

            with gr.Column(scale=2):
                gr.Markdown("Choose the Language Model")
                model_choices = ragsst.list_local_models()
                model_name = gr.Dropdown(
                    info="Choose a locally available LLM to use",
                    choices=model_choices,
                    allow_custom_value=True,
                    value=MODEL,
                    label="Local LLM",
                    interactive=True,
                )

                setllm_btn = gr.Button("Set Choice", size='sm')
                setllm_btn.click(fn=ragsst.set_model, inputs=model_name)

                pull_model_name = gr.Dropdown(
                    info="Download a LLM (Internet connection is required)",
                    choices=p.LLM_CHOICES,
                    allow_custom_value=True,
                    value=MODEL,
                    label="LLM",
                    interactive=True,
                )
                setllm_btn = gr.Button("Download", size='sm')
                pull_info = gr.Textbox(label="Info")
                setllm_btn.click(fn=ragsst.pull_model, inputs=pull_model_name, outputs=pull_info)

                def update_local_models_list(progress_info):
                    if "success" in progress_info.lower():
                        return gr.Dropdown(
                            choices=ragsst.list_local_models(), value=MODEL, interactive=True
                        )
                    return model_name

                pull_info.change(update_local_models_list, pull_info, model_name)

    gui = gr.TabbedInterface(
        [rag_query_ui, semantic_retrieval_ui, rag_chat_ui, chat_ui, config_ui],
        ["RAG Query", "Semantic Retrieval", "RAG Chat", "Chat", "Rag Tool Settings"],
        title="<a href='https://github.com/aihpi/ragsst' target='_blank'>Local RAG Tool</a>",
    )

    return gui
