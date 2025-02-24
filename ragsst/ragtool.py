import os
import logging
import chromadb
from chromadb.utils import embedding_functions
import requests, json
from tqdm import tqdm
from typing import List, Any, Generator, Deque
from collections import deque
from ragsst.utils import list_files, read_file, split_text, hash_file
from yake import KeywordExtractor
import ragsst.parameters as p


logging.basicConfig(format=os.getenv('LOG_FORMAT', '%(asctime)s [%(levelname)s] %(message)s'))
logger = logging.getLogger(__name__)
logger.setLevel(os.getenv('LOG_LEVEL', logging.INFO))
logger.addHandler(logging.FileHandler(os.path.join(p.LOG_DIR, p.LOG_FILE), mode='w+'))

# Assign default values
MODEL = p.LLM_CHOICES[0]
EMBEDDING_MODEL = p.EMBEDDING_MODELS[0]


class RAGTool:
    def __init__(
        self,
        model: str = MODEL,
        llm_base_url: str = p.LLMBASEURL,
        data_path: str = p.DATA_PATH,
        embedding_model: str = EMBEDDING_MODEL,
        collection_name: str = p.COLLECTION_NAME,
    ):
        self.model = model
        self.llm_base_url = llm_base_url
        self.max_conversation_length = p.CONVERSATION_LENTGH
        self.conversation = deque(maxlen=self.max_conversation_length)
        self.rag_conversation = deque(maxlen=self.max_conversation_length)
        self.data_path = data_path
        self.embedding_model = embedding_model
        self.collection_name = collection_name
        self.vs_client = chromadb.PersistentClient(
            path=p.VECTOR_DB_PATH, settings=chromadb.Settings(allow_reset=True)
        )
        self.embedding_func = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=self.embedding_model, trust_remote_code=True
        )
        if p.KEYWORD_SEARCH or p.FILTER_BY_KEYWORD:
            self.kw_extractor = KeywordExtractor(
                lan="auto",
                n=1,
                dedupLim=0.9,
                windowsSize=1,
                top=1,
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
            response = response_dic.get('response', '')
            return response if response else response_dic.get('error', 'Check Ollama Settings')

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

    def pull_model(self, model_name) -> Generator[str, str, None]:

        url = self.llm_base_url + "/pull"

        data = {"name": model_name}

        try:
            r = requests.post(url, json=data, stream=True)
            r.raise_for_status()
            for content in r.iter_lines():
                if content:
                    content_dict = json.loads(content)
                    yield f"Status: {content_dict.get('status')}"

        except Exception as e:
            logger.error(f"Exception: {e}\nResponse:{r}")

    # ============== Vector Store ==============================================

    def set_collection(self, collection_name: str, embedding_model: str = None) -> None:
        self.set_collection_name(collection_name)
        if embedding_model is not None:
            self.set_embeddings_model(embedding_model)
        self.collection = self.vs_client.get_or_create_collection(
            name=self.collection_name,
            embedding_function=self.embedding_func,
            metadata={"hnsw:space": "cosine", "embedding_model": self.embedding_model},
        )
        logger.info(
            f"Set Collection: {self.collection_name}. Embedding Model: {self.embedding_model}"
        )

    def make_collection(
        self,
        data_path: str,
        collection_name: str,
        skip_included_files: bool = True,
        consider_content: bool = True,
    ) -> None:
        """Create vector store collection from a set of documents"""

        logger.info(f"Documents Path: {data_path}")

        self.set_collection(collection_name, None)

        files = list_files(data_path, extensions=('.txt', '.pdf', '.docx'))
        logger.info(f"{len(files)} files found.")
        logger.debug(f"Files: {', '.join([f.replace(data_path, '', 1) for f  in files])}")
        logger.info("Populating embeddings database...")

        if skip_included_files:
            sources = {
                m.get('source')
                for m in self.collection.get(include=['metadatas']).get('metadatas')
            }
            if consider_content:
                files_hashes = {
                    m.get('file_hash')
                    for m in self.collection.get(include=['metadatas']).get('metadatas')
                }

        for f in files:
            _, file_name = os.path.split(f)
            if consider_content:
                file_hash = hash_file(f)

            if skip_included_files and file_name in sources:
                if not consider_content:
                    logger.info(f"{file_name} name already in Vector-DB, skipping...")
                    continue

                if file_hash in files_hashes:
                    logger.info(f"{file_name} content already in Vector-DB, skipping...")
                    continue

                logger.info(f"Updating DB for {file_name} ...")
                self.collection.delete(where={"source": file_name})

            logger.info(f"Reading and splitting {file_name} ...")
            text = read_file(f)
            chunks = split_text(text)
            logger.info(f"Resulting segment count: {len(chunks)}")
            logger.info(f"Embedding and storing {file_name} ...")

            for i, c in tqdm(enumerate(chunks, 1), total=len(chunks)):
                metadata = {"source": file_name, "part": i}
                if consider_content:
                    metadata["file_hash"] = file_hash

                self.collection.add(
                    documents=c,
                    ids=f"id{file_name[:-4]}.{i}",
                    metadatas=metadata,
                )

        logger.info(f"Available collections: {self.list_collections_names_w_metainfo()}")

    # ============== Semantic Search / Retrieval ===============================

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
            metadata = query_result.get('metadatas')[0][i]
            docs_selection.append(
                '\n'.join(
                    [
                        doc,
                        f"Relevance: {sim}",
                        f"Source: {metadata.get('source')} (part {metadata.get('part')})",
                    ]
                )
            )

        if not docs_selection:
            return "Relevant passage not found. Try lowering the relevance threshold."

        return "\n-----------------\n\n".join(docs_selection)

    def get_relevant_text(
        self,
        query: str = '',
        nresults: int = 2,
        sim_th: float | None = None,
        keyword_filter: bool = p.FILTER_BY_KEYWORD,
        keyword_search: bool = p.KEYWORD_SEARCH,
    ) -> str:
        """Get relevant text from a collection for a given query"""

        query_result = self.collection.query(query_texts=query, n_results=nresults)

        if sim_th is not None:
            # Filter documents based on similarity threshold
            filtered_query = self._filter_query_by_similarity(query_result, sim_th)

            if filtered_query:
                if keyword_filter:
                    # Extract the main keyword from the query
                    kw = self.kw_extractor.extract_keywords(query)[0][0]
                    # Filter relevant documents based on the extracted keyword
                    kw_filtered_query = self._filter_query_by_keyword(filtered_query, kw)

                    if kw_filtered_query:
                        logger.debug("Semantic retrieval succesfully filtered by keyword")
                        filtered_query = kw_filtered_query

                query_result = filtered_query

            # If no relevant documents found after previous criterias perform keyword search if enabled
            elif keyword_search:
                kw = self.kw_extractor.extract_keywords(query)[0][0]
                logger.debug(f"No results by semantic search. Searching by Keyword: {kw}")
                query_result = self.collection.query(
                    query_texts="", n_results=nresults, where_document={"$contains": kw}
                )

            else:
                logger.info("No results by semantic search")
                return ""

        relevant_docs = query_result.get('documents')[0]
        if relevant_docs:
            logger.info(f"Sources:  {', '.join(self._get_sources(query_result))}")
        return '\n'.join(relevant_docs)

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

    def get_condenser_prompt(self, query: str, chat_history: Deque) -> str:
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
        if not relevant_text:
            return "Relevant passage not found. Try lowering the relevance threshold."
        logger.debug(f"\nSelected Relevant Context:\n{relevant_text}")

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
        MSG_NO_CONTEXT = "Relevant passage not found. Try lowering the relevance threshold."

        if not self.rag_conversation:
            relevant_text = self.get_relevant_text(user_msg, nresults=nresults, sim_th=sim_th)
            if not relevant_text:
                return MSG_NO_CONTEXT
            logger.debug(f"\nSelected Relevant Context:\n{relevant_text}")
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
        logger.debug(f"\nPassed Relevant Context:\n{relevant_text}")
        contextualized_standalone_query = self.get_context_prompt(standalone_query, relevant_text)

        bot_response = self.llm_generate(
            contextualized_standalone_query, top_k=top_k, top_p=top_p, temp=temp
        )
        self.rag_conversation.append('Query:\n' + standalone_query)
        self.rag_conversation.append('Answer:\n' + bot_response)
        return bot_response

    # ============== LLM chat w/o Document Context =============================

    def chat(self, user_msg: str, history: Any, top_k: int, top_p: float, temp: float) -> str:
        bot_response = self.llm_chat(user_msg, top_k=top_k, top_p=top_p, temp=temp)
        return bot_response

    # ============== Utils =====================================================
    # Methods for internal usage and/or interaction with the GUI

    def _check_initdb_conditions(self) -> bool:

        return (
            os.path.exists(self.data_path)
            and os.listdir(self.data_path)
            and (
                not os.path.exists(p.VECTOR_DB_PATH)
                or not [f.path for f in os.scandir(p.VECTOR_DB_PATH) if f.is_dir()]
            )
        )

    def setup_vec_store(self, collection_name: str = p.COLLECTION_NAME) -> None:
        "Vector Store Initialization Setup"

        if self._check_initdb_conditions():
            logger.debug("Init DB contitions are met")
            self.make_collection(self.data_path, collection_name)
        else:
            collections = self.vs_client.list_collections()
            if collections:
                logger.info(f"Available collections: {self.list_collections_names_w_metainfo()}")
                self.set_collection(
                    collections[0].name, collections[0].metadata.get("embedding_model")
                )
                if not self.collection.peek(limit=1).get("ids"):
                    logger.info("The Set Collection is empty. Populate it or choose another one")
            else:
                self.set_collection(collection_name)
                logger.warning("The Database is empty. Make/Update Database")

    def set_model(self, llm: str) -> None:
        self.model = llm
        logger.info(f"Chosen Model: {self.model}")

    def set_embeddings_model(self, emb_model: str) -> None:
        self.embedding_model = emb_model
        self.embedding_func = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=self.embedding_model,
            trust_remote_code=True,
        )
        logger.debug(f"Embedding Model: {self.embedding_model}")

    def set_data_path(self, data_path: str) -> None:
        self.data_path = data_path
        logger.debug(f"Data Path: {self.data_path}")

    def set_collection_name(self, collection_name: str) -> None:
        self.collection_name = collection_name
        logger.debug(f"Collection Name: {self.collection_name}")

    def list_collections_names(self) -> List:
        return [c.name for c in self.vs_client.list_collections()]

    def list_collections_names_w_metainfo(self) -> str:
        return ', '.join(
            [
                f"{c.name} ({c.metadata.get('embedding_model','')})"
                for c in self.vs_client.list_collections()
            ]
        )

    def delete_collection(self, collection_name: str) -> None:
        """Removes chosen collection and sets the first one on the list"""
        self.vs_client.delete_collection(collection_name)
        logger.info(f"{collection_name} removed")
        collections = self.vs_client.list_collections()
        if collections:
            logger.info(f"Setting first available collection: {collections[0].name}")
            self.set_collection(
                collections[0].name, collections[0].metadata.get("embedding_model")
            )

    def clean_database(self) -> None:
        """Deletes all collections and entries"""
        self.vs_client.reset()
        self.vs_client.clear_system_cache()
        logger.info("Database empty")

    def clear_chat_hist(self) -> None:
        self.conversation.clear()

    def clear_ragchat_hist(self) -> None:
        self.rag_conversation.clear()

    def filter_strings(self, docs: List, keyword: str) -> List[str]:
        logger.debug(f"Chosen Keyword: {keyword}")
        keyword = keyword.lower()
        return [s for s in docs if keyword in s.lower()]

    def _filter_by_similarity(self, query_result: dict, sim_th: float) -> List[str]:
        """Filter documents based on similarity threshold and return relevant docs"""
        similarities = [1 - d for d in query_result.get('distances')[0]]
        relevant_docs = [
            doc for doc, s in zip(query_result.get('documents')[0], similarities) if s >= sim_th
        ]
        return relevant_docs

    def _filter_query_by_similarity(self, query_result: dict, sim_th: float) -> dict:
        """Filter query results based on similarity threshold."""
        similarities = [round(1 - d, 2) for d in query_result.get('distances')[0]]
        relevant_docs = [
            doc for doc, s in zip(query_result.get('documents')[0], similarities) if s >= sim_th
        ]
        if relevant_docs:
            metadatas = [
                meta
                for meta, s in zip(query_result.get('metadatas')[0], similarities)
                if s >= sim_th
            ]
            query_result['documents'][0] = relevant_docs
            query_result['metadatas'][0] = metadatas
            return query_result
        return {}

    def _filter_query_by_keyword(self, query_result: dict, keyword: str) -> dict:
        """Filter query results based on keyword."""
        logger.debug(f"Chosen Keyword: {keyword}")
        keyword = keyword.lower()
        relevant_docs = [doc for doc in query_result.get('documents')[0] if keyword in doc.lower()]
        if relevant_docs:
            metadatas = [
                meta
                for meta, doc in zip(
                    query_result.get('metadatas')[0], query_result.get('documents')[0]
                )
                if keyword in doc.lower()
            ]
            query_result['documents'][0] = relevant_docs
            query_result['metadatas'][0] = metadatas
            return query_result
        return {}

    def _get_sources(self, query_result: dict) -> set:
        """Get sources from the query results."""
        return {meta.get("source") for meta in query_result['metadatas'][0]}
