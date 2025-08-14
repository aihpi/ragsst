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

    # Multi Query RAG Interface with three-column layout
    with gr.Blocks() as multi_query_ui:
        gr.Markdown("## Multi Query RAG")
        gr.Markdown("Generate multiple paraphrases of your query to retrieve more comprehensive results.")
        
        with gr.Row():
            # Column 1: Input and Parameters
            with gr.Column(scale=1):
                gr.Markdown("### Query & Parameters")
                query_input = gr.Textbox(
                    label="Your Query",
                    placeholder="Enter your question here...",
                    lines=3
                )
                
                with gr.Accordion("Settings", open=False):
                    num_paraphrases = gr.Slider(
                        1, 5, value=3, step=1, 
                        label="Number of Paraphrases",
                        info="How many query variations to generate"
                    )
                    relevance_threshold = gr.Slider(
                        0, 1, value=0.5, step=0.1,
                        label="Relevance threshold", 
                        info=pinfo.get("Rth")
                    )
                    top_n_results = gr.Slider(
                        1, 5, value=3, step=1,
                        label="Top n results", 
                        info=pinfo.get("TopnR")
                    )
                    top_k = gr.Slider(
                        1, 10, value=5, step=1,
                        label="Top k", 
                        info=pinfo.get("Top k")
                    )
                    top_p = gr.Slider(
                        0.1, 1, value=0.9, step=0.1,
                        label="Top p", 
                        info=pinfo.get("Top p"),
                        visible=False
                    )
                    temp = gr.Slider(
                        0.1, 1, value=0.3, step=0.1,
                        label="Temp", 
                        info=pinfo.get("Temp")
                    )
                
                submit_btn = gr.Button("Generate Multi Query Response", variant="primary", size="lg")
                clear_btn = gr.Button("Clear", size="sm")
            
            # Column 2: Query Results and Retrieved Documents
            with gr.Column(scale=2):
                gr.Markdown("### Queries & Retrieved Documents")
                query_results_display = gr.Markdown(
                    value="*Submit a query to see paraphrases and retrieved documents*",
                    label="Query Results"
                )
            
            # Column 3: Final Response
            with gr.Column(scale=1):
                gr.Markdown("### Final Response")
                final_response_display = gr.Textbox(
                    value="",
                    label="Generated Answer",
                    lines=20,
                    interactive=False
                )
        
        # Function to handle multi query RAG
        def handle_multi_query(query, num_para, rel_th, top_n, top_k_val, top_p_val, temp_val):
            if not query.strip():
                return "Please enter a query.", "*Enter a query to see results*"
            
            try:
                final_response, query_results, formatted_results = ragsst.multi_query_rag(
                    user_msg=query,
                    sim_th=rel_th,
                    nresults=top_n,
                    top_k=top_k_val,
                    top_p=top_p_val,
                    temp=temp_val,
                    num_paraphrases=num_para
                )
                return formatted_results, final_response
            except Exception as e:
                error_msg = f"Error: {str(e)}"
                return error_msg, error_msg
        
        def clear_multi_query():
            return "", "*Submit a query to see paraphrases and retrieved documents*", ""
        
        # Event handlers
        submit_btn.click(
            fn=handle_multi_query,
            inputs=[query_input, num_paraphrases, relevance_threshold, top_n_results, top_k, top_p, temp],
            outputs=[query_results_display, final_response_display]
        )
        
        clear_btn.click(
            fn=clear_multi_query,
            outputs=[query_input, query_results_display, final_response_display]
        )

    # RAG Fusion Interface with three-column layout
    with gr.Blocks() as rag_fusion_ui:
        gr.Markdown("## RAG Fusion")
        gr.Markdown("Generate multiple paraphrases, apply weighted fusion scoring, and rank documents for comprehensive results.")
        
        with gr.Row():
            # Column 1: Input and Parameters
            with gr.Column(scale=1):
                gr.Markdown("### Query & Parameters")
                fusion_query_input = gr.Textbox(
                    label="Your Query",
                    placeholder="Enter your question here...",
                    lines=3
                )
                
                with gr.Accordion("Settings", open=False):
                    fusion_num_paraphrases = gr.Slider(
                        1, 5, value=3, step=1, 
                        label="Number of Paraphrases",
                        info="How many query variations to generate"
                    )
                    fusion_method = gr.Dropdown(
                        choices=["max_score", "weighted_avg", "boost_multiple", "top_n_only"],
                        value="max_score",
                        label="Fusion Method",
                        info="How to combine scores from multiple queries"
                    )
                    fusion_relevance_threshold = gr.Slider(
                        0, 1, value=0.5, step=0.1,
                        label="Relevance threshold", 
                        info=pinfo.get("Rth")
                    )
                    fusion_score_threshold = gr.Slider(
                        0, 1, value=0.3, step=0.1,
                        label="Fusion Score Threshold",
                        info="Minimum fusion score to include document (ignored for 'top_n_only')"
                    )
                    fusion_top_n_results = gr.Slider(
                        1, 5, value=3, step=1,
                        label="Top n results", 
                        info=pinfo.get("TopnR")
                    )
                    fusion_top_k = gr.Slider(
                        1, 10, value=5, step=1,
                        label="Top k", 
                        info=pinfo.get("Top k")
                    )
                    fusion_top_p = gr.Slider(
                        0.1, 1, value=0.9, step=0.1,
                        label="Top p", 
                        info=pinfo.get("Top p"),
                        visible=False
                    )
                    fusion_temp = gr.Slider(
                        0.1, 1, value=0.3, step=0.1,
                        label="Temp", 
                        info=pinfo.get("Temp")
                    )
                
                fusion_submit_btn = gr.Button("Generate RAG Fusion Response", variant="primary", size="lg")
                fusion_clear_btn = gr.Button("Clear", size="sm")
            
            # Column 2: Query Results and Retrieved Documents + Filtering Info
            with gr.Column(scale=2):
                gr.Markdown("### Queries & Retrieved Documents")
                fusion_query_results_display = gr.Markdown(
                    value="*Submit a query to see paraphrases and retrieved documents*",
                    label="Query Results"
                )
                
                gr.Markdown("### Fusion Filtering Details")
                fusion_filtering_display = gr.Markdown(
                    value="*Filtering information will appear here after processing*",
                    label="Filtering Process"
                )
            
            # Column 3: Final Response
            with gr.Column(scale=1):
                gr.Markdown("### Final Response")
                fusion_final_response_display = gr.Textbox(
                    value="",
                    label="Generated Answer",
                    lines=20,
                    interactive=False
                )
        
        # Function to handle RAG fusion
        def handle_rag_fusion(query, num_para, fusion_method_val, rel_th, fusion_th, top_n, top_k_val, top_p_val, temp_val):
            if not query.strip():
                return "Please enter a query.", "*Enter a query to see results*", "*No filtering performed*"
            
            try:
                final_response, query_results, formatted_results, filtering_info = ragsst.rag_fusion(
                    user_msg=query,
                    sim_th=rel_th,
                    nresults=top_n,
                    top_k=top_k_val,
                    top_p=top_p_val,
                    temp=temp_val,
                    num_paraphrases=num_para,
                    fusion_threshold=fusion_th,
                    fusion_method=fusion_method_val
                )
                return formatted_results, filtering_info, final_response
            except Exception as e:
                error_msg = f"Error: {str(e)}"
                return error_msg, error_msg, error_msg
        
        def clear_rag_fusion():
            return "", "*Submit a query to see paraphrases and retrieved documents*", "*Filtering information will appear here after processing*", ""
        
        # Event handlers
        fusion_submit_btn.click(
            fn=handle_rag_fusion,
            inputs=[fusion_query_input, fusion_num_paraphrases, fusion_method, fusion_relevance_threshold, 
                   fusion_score_threshold, fusion_top_n_results, fusion_top_k, fusion_top_p, fusion_temp],
            outputs=[fusion_query_results_display, fusion_filtering_display, fusion_final_response_display]
        )
        
        fusion_clear_btn.click(
            fn=clear_rag_fusion,
            outputs=[fusion_query_input, fusion_query_results_display, fusion_filtering_display, fusion_final_response_display]
        )

    # ================= Decomposition RAG UI =======================
    with gr.Blocks() as decomposition_ui:
        gr.Markdown("## Decomposition RAG (Least-to-Most Prompting)")
        gr.Markdown("Break down complex queries into simpler subproblems, solve sequentially or independently, then synthesize the final answer.")
        
        with gr.Row():
            # Column 1: Input and Parameters
            with gr.Column(scale=1):
                gr.Markdown("### Query & Parameters")
                decomp_query_input = gr.Textbox(
                    label="Your Complex Query",
                    placeholder="Enter a complex question that can be broken down into parts...",
                    lines=3
                )
                
                with gr.Accordion("Settings", open=False):
                    decomp_max_subqueries = gr.Slider(
                        2, 6, value=4, step=1, 
                        label="Max Subqueries",
                        info="Maximum number of subquestions to generate"
                    )
                    decomp_method = gr.Dropdown(
                        choices=["least_to_most", "independent"],
                        value="least_to_most",
                        label="Decomposition Method",
                        info="Sequential (builds on previous) vs Independent processing"
                    )
                    decomp_relevance_threshold = gr.Slider(
                        0, 1, value=0.5, step=0.1,
                        label="Relevance threshold", 
                        info=pinfo.get("Rth")
                    )
                    decomp_top_n_results = gr.Slider(
                        1, 5, value=3, step=1,
                        label="Top n results", 
                        info=pinfo.get("TopnR")
                    )
                    decomp_top_k = gr.Slider(
                        1, 10, value=5, step=1,
                        label="Top k", 
                        info=pinfo.get("Top k")
                    )
                    decomp_top_p = gr.Slider(
                        0.1, 1, value=0.9, step=0.1,
                        label="Top p", 
                        info=pinfo.get("Top p"),
                        visible=False
                    )
                    decomp_temp = gr.Slider(
                        0.1, 1, value=0.3, step=0.1,
                        label="Temp", 
                        info=pinfo.get("Temp")
                    )
                
                decomp_submit_btn = gr.Button("Generate Decomposition Response", variant="primary", size="lg")
                decomp_clear_btn = gr.Button("Clear", size="sm")
            
            # Column 2: Subqueries and Step-by-Step Results
            with gr.Column(scale=2):
                gr.Markdown("### Subqueries & Step-by-Step Results")
                decomp_step_results_display = gr.Markdown(
                    value="*Submit a query to see decomposition and step-by-step processing*",
                    label="Step Results"
                )
                
                gr.Markdown("### Decomposition Process Details")
                decomp_process_display = gr.Markdown(
                    value="*Decomposition information will appear here after processing*",
                    label="Process Details"
                )
            
            # Column 3: Final Response
            with gr.Column(scale=1):
                gr.Markdown("### Final Synthesized Response")
                decomp_final_response_display = gr.Textbox(
                    value="",
                    label="Generated Answer",
                    lines=20,
                    interactive=False
                )
        
        # Function to handle decomposition RAG
        def handle_decomposition_rag(query, max_sub, decomp_method_val, rel_th, top_n, top_k_val, top_p_val, temp_val):
            if not query.strip():
                return "Please enter a query.", "*Enter a query to see results*", "*No decomposition performed*"
            
            try:
                final_response, query_results, formatted_results, decomposition_info = ragsst.decomposition_rag(
                    user_msg=query,
                    sim_th=rel_th,
                    nresults=top_n,
                    top_k=top_k_val,
                    top_p=top_p_val,
                    temp=temp_val,
                    max_subqueries=max_sub,
                    decomposition_method=decomp_method_val
                )
                return formatted_results, decomposition_info, final_response
            except Exception as e:
                error_msg = f"Error: {str(e)}"
                return error_msg, error_msg, error_msg
        
        def clear_decomposition():
            return "", "*Submit a query to see decomposition and step-by-step processing*", "*Decomposition information will appear here after processing*", ""
        
        # Event handlers
        decomp_submit_btn.click(
            fn=handle_decomposition_rag,
            inputs=[decomp_query_input, decomp_max_subqueries, decomp_method, decomp_relevance_threshold, 
                   decomp_top_n_results, decomp_top_k, decomp_top_p, decomp_temp],
            outputs=[decomp_step_results_display, decomp_process_display, decomp_final_response_display]
        )
        
        decomp_clear_btn.click(
            fn=clear_decomposition,
            outputs=[decomp_query_input, decomp_step_results_display, decomp_process_display, decomp_final_response_display]
        )

    # ================= HyDE RAG UI =======================
    with gr.Blocks() as hyde_ui:
        gr.Markdown("## HyDE RAG (Hypothetical Document Embeddings)")
        gr.Markdown("Generate hypothetical documents that would answer your query, then use them for enhanced embedding-based retrieval.")
        
        with gr.Row():
            # Column 1: Input and Parameters
            with gr.Column(scale=1):
                gr.Markdown("### Query & Parameters")
                hyde_query_input = gr.Textbox(
                    label="Your Query",
                    placeholder="Enter your question here...",
                    lines=3
                )
                
                with gr.Accordion("Settings", open=False):
                    hyde_num_hypotheses = gr.Slider(
                        1, 3, value=1, step=1, 
                        label="Number of Hypotheses",
                        info="How many hypothetical documents to generate"
                    )
                    hyde_hypothesis_length = gr.Dropdown(
                        choices=["short", "paragraph", "detailed"],
                        value="paragraph",
                        label="Hypothesis Length",
                        info="Length of generated hypothetical documents"
                    )
                    hyde_relevance_threshold = gr.Slider(
                        0, 1, value=0.5, step=0.1,
                        label="Relevance threshold", 
                        info=pinfo.get("Rth")
                    )
                    hyde_top_n_results = gr.Slider(
                        1, 5, value=3, step=1,
                        label="Top n results", 
                        info=pinfo.get("TopnR")
                    )
                    hyde_top_k = gr.Slider(
                        1, 10, value=5, step=1,
                        label="Top k", 
                        info=pinfo.get("Top k")
                    )
                    hyde_top_p = gr.Slider(
                        0.1, 1, value=0.9, step=0.1,
                        label="Top p", 
                        info=pinfo.get("Top p"),
                        visible=False
                    )
                    hyde_temp = gr.Slider(
                        0.1, 1, value=0.3, step=0.1,
                        label="Temp", 
                        info=pinfo.get("Temp")
                    )
                
                hyde_submit_btn = gr.Button("Generate HyDE Response", variant="primary", size="lg")
                hyde_clear_btn = gr.Button("Clear", size="sm")
            
            # Column 2: Hypotheses and Retrieved Documents
            with gr.Column(scale=2):
                gr.Markdown("### Hypotheses & Retrieved Documents")
                hyde_hypotheses_display = gr.Markdown(
                    value="*Submit a query to see generated hypotheses and retrieved documents*",
                    label="Hypotheses Results"
                )
                
                gr.Markdown("### HyDE Process Details")
                hyde_process_display = gr.Markdown(
                    value="*HyDE process information will appear here after processing*",
                    label="Process Details"
                )
            
            # Column 3: Final Response
            with gr.Column(scale=1):
                gr.Markdown("### Final Response")
                hyde_final_response_display = gr.Textbox(
                    value="",
                    label="Generated Answer",
                    lines=20,
                    interactive=False
                )
        
        # Function to handle HyDE RAG
        def handle_hyde_rag(query, num_hyp, hyp_length, rel_th, top_n, top_k_val, top_p_val, temp_val):
            if not query.strip():
                return "Please enter a query.", "*Enter a query to see results*", "*No HyDE processing performed*"
            
            try:
                final_response, query_results, formatted_results, hyde_info = ragsst.hyde_rag(
                    user_msg=query,
                    sim_th=rel_th,
                    nresults=top_n,
                    top_k=top_k_val,
                    top_p=top_p_val,
                    temp=temp_val,
                    num_hypotheses=num_hyp,
                    hypothesis_length=hyp_length
                )
                return formatted_results, hyde_info, final_response
            except Exception as e:
                error_msg = f"Error: {str(e)}"
                return error_msg, error_msg, error_msg
        
        def clear_hyde():
            return "", "*Submit a query to see generated hypotheses and retrieved documents*", "*HyDE process information will appear here after processing*", ""
        
        # Event handlers
        hyde_submit_btn.click(
            fn=handle_hyde_rag,
            inputs=[hyde_query_input, hyde_num_hypotheses, hyde_hypothesis_length, hyde_relevance_threshold, 
                   hyde_top_n_results, hyde_top_k, hyde_top_p, hyde_temp],
            outputs=[hyde_hypotheses_display, hyde_process_display, hyde_final_response_display]
        )
        
        hyde_clear_btn.click(
            fn=clear_hyde,
            outputs=[hyde_query_input, hyde_hypotheses_display, hyde_process_display, hyde_final_response_display]
        )

    gui = gr.TabbedInterface(
        [rag_query_ui, semantic_retrieval_ui, rag_chat_ui, chat_ui, multi_query_ui, rag_fusion_ui, decomposition_ui, hyde_ui, config_ui],
        ["RAG Query", "Semantic Retrieval", "RAG Chat", "Chat", "Multi Query RAG", "RAG Fusion", "Decomposition RAG", "HyDE RAG", "Rag Tool Settings"],
        title="<a href='https://github.com/aihpi/ragsst' target='_blank'>Local RAG Tool</a>",
    )

    return gui
