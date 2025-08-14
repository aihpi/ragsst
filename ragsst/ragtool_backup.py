import os
import logging
import chromadb
from chromadb.utils import embedding_functions
import requests, json
from tqdm import tqdm
from typing import List, Any, Generator, Deque
from collections import deque
from ragsst.utils import list_files, read_file, split_text, hash_file
from ragsst.query_transformations import QueryTransformations
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
        #print("RAGTool __init__ started")
        self.model = model
        self.llm_base_url = llm_base_url
        self.max_conversation_length = p.CONVERSATION_LENTGH
        self.conversation = deque(maxlen=self.max_conversation_length)
        self.rag_conversation = deque(maxlen=self.max_conversation_length)
        self.data_path = data_path
        self.embedding_model = embedding_model
        self.collection_name = collection_name
        #print("Before chromadb.PersistentClient")
        self.vs_client = chromadb.PersistentClient(
            path=p.VECTOR_DB_PATH, settings=chromadb.Settings(allow_reset=True)
        )
        print("Loading sentence transformer model for embeddings")
        self.embedding_func = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=self.embedding_model, trust_remote_code=True
        )
        print("Succesfully loaded sentence transformer model for embeddings")
        if p.KEYWORD_SEARCH or p.FILTER_BY_KEYWORD:
            #print("Before KeywordExtractor")
            self.kw_extractor = KeywordExtractor(
                lan="auto",
                n=1,
                dedupLim=0.9,
                windowsSize=1,
                top=1,
            )
            #print("After KeywordExtractor")
        #print("RAGTool __init__ finished")

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
        
        # Initialize response_dic to None to avoid UnboundLocalError
        response_dic = {}
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

        response_dic = {}
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
        response_dic = {}

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
        """Create vector store collection from a set of documents with incremental updates"""

        logger.info(f"Documents Path: {data_path}")

        self.set_collection(collection_name, None)

        files = list_files(data_path, extensions=('.txt', '.pdf', '.docx'))
        logger.info(f"{len(files)} files found.")
        logger.debug(f"Files: {', '.join([f.replace(data_path, '', 1) for f  in files])}")
        
        # Get current files in data directory
        current_files = {os.path.basename(f) for f in files}
        
        # Get existing files in database
        existing_data = self.collection.get(include=['metadatas'])
        existing_files = {m.get('source') for m in existing_data.get('metadatas', [])}
        existing_hashes = {}
        
        if consider_content:
            # Build a mapping of filename to hash for existing files
            for metadata in existing_data.get('metadatas', []):
                source = metadata.get('source')
                file_hash = metadata.get('file_hash')
                if source and file_hash:
                    existing_hashes[source] = file_hash

        # Remove files from database that no longer exist in data directory
        files_to_remove = existing_files - current_files
        if files_to_remove:
            logger.info(f"Removing {len(files_to_remove)} files no longer in data directory...")
            for file_to_remove in files_to_remove:
                logger.info(f"Removing {file_to_remove} from database...")
                self.collection.delete(where={"source": file_to_remove})

        logger.info("Populating embeddings database...")

        for f in files:
            _, file_name = os.path.split(f)
            current_hash = None
            
            if consider_content:
                current_hash = hash_file(f)

            # Check if file should be processed
            if skip_included_files and file_name in existing_files:
                if not consider_content:
                    logger.info(f"{file_name} already in Vector-DB, skipping...")
                    continue

                # Check if content has changed
                if consider_content and current_hash == existing_hashes.get(file_name):
                    logger.info(f"{file_name} content unchanged, skipping...")
                    continue

                # File exists but content changed - update it
                logger.info(f"Content changed for {file_name}, updating database...")
                self.collection.delete(where={"source": file_name})
            else:
                logger.info(f"Processing new file: {file_name}...")

            # Process the file (new or updated)
            logger.info(f"Reading and splitting {file_name}...")
            text = read_file(f)
            chunks = split_text(text)
            logger.info(f"Resulting segment count: {len(chunks)}")
            logger.info(f"Embedding and storing {file_name}...")

            for i, c in tqdm(enumerate(chunks, 1), total=len(chunks)):
                metadata = {"source": file_name, "part": i}
                if consider_content:
                    metadata["file_hash"] = current_hash

                self.collection.add(
                    documents=c,
                    ids=f"id{file_name[:-4]}.{i}",
                    metadatas=metadata,
                )

        logger.info(f"Available collections: {self.list_collections_names_w_metainfo()}")

    def sync_collection_with_data(
        self,
        data_path: str = None,
        collection_name: str = None,
        consider_content: bool = True,
    ) -> None:
        """
        Synchronize the vector database with the current state of the data directory.
        This is a convenience method that calls make_collection with sync behavior.
        """
        if data_path is None:
            data_path = self.data_path
        if collection_name is None:
            collection_name = self.collection_name
            
        logger.info("Synchronizing vector database with data directory...")
        self.make_collection(
            data_path=data_path,
            collection_name=collection_name,
            skip_included_files=True,
            consider_content=consider_content,
        )

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

    # ============== Query Transformations ====================================

    def multi_query_rag(
        self, 
        user_msg: str, 
        sim_th: float, 
        nresults: int, 
        top_k: int, 
        top_p: float, 
        temp: float,
        num_paraphrases: int = 3
    ) -> tuple[str, list[dict], str]:
        """
        Multi Query RAG: Generate multiple paraphrases, retrieve for each, then generate response
        Returns: (final_response, query_results, formatted_results)
        """
        return QueryTransformations.multi_query_rag(
            self, user_msg, sim_th, nresults, top_k, top_p, temp, num_paraphrases
        )

    def rag_fusion(
        self, 
        user_msg: str, 
        sim_th: float, 
        nresults: int, 
        top_k: int, 
        top_p: float, 
        temp: float,
        num_paraphrases: int = 3,
        fusion_threshold: float = 0.5,
        fusion_method: str = "max_score"
    ) -> tuple[str, list[dict], str, str]:
        """
        RAG Fusion: Generate paraphrases, retrieve for each, apply weighted fusion, then generate response
        Returns: (final_response, query_results, formatted_results, filtering_info)
        """
        return QueryTransformations.rag_fusion(
            self, user_msg, sim_th, nresults, top_k, top_p, temp, 
            num_paraphrases, fusion_threshold, fusion_method
        )

    def decomposition_rag(
        self, 
        user_msg: str, 
        sim_th: float, 
        nresults: int, 
        top_k: int, 
        top_p: float, 
        temp: float,
        max_subqueries: int = 4,
        decomposition_method: str = "least_to_most"
    ) -> tuple[str, list[dict], str, str]:
        """
        Decomposition RAG: Break down complex queries into simpler subproblems
        Returns: (final_response, query_results, formatted_results, decomposition_info)
        """
        return QueryTransformations.decomposition_rag(
            self, user_msg, sim_th, nresults, top_k, top_p, temp,
            max_subqueries, decomposition_method
        )

    # ============== Retrieval Augemented Generation ===========================

    def multi_query_rag(
        self, 
        user_msg: str, 
        sim_th: float, 
        nresults: int, 
        top_k: int, 
        top_p: float, 
        temp: float,
        num_paraphrases: int = 3
    ) -> tuple[str, list[dict], str]:
        """
        Multi Query RAG: Generate multiple paraphrases, retrieve for each, then generate response
        Returns: (final_response, query_results, formatted_results)
        """
        logger.debug(f"multi_query_rag args: sim_th: {sim_th}, nresults: {nresults}, num_paraphrases: {num_paraphrases}")
        
        # Step 1: Generate paraphrases of the original query
        paraphrase_prompt = f"""Please provide {num_paraphrases} different ways to ask the same question. Each version should:
- Ask for the same information as the original
- Use different words and sentence structure
- NOT add any assumptions, context, or specificity that wasn't in the original
- Keep the same level of generality as the original question

Original question: {user_msg}

Provide only the alternative questions, one per line:"""
        
        paraphrases_response = self.llm_generate(paraphrase_prompt, top_k=top_k, top_p=top_p, temp=0.7)
        
        # Parse paraphrases
        paraphrases = [line.strip() for line in paraphrases_response.split('\n') if line.strip()]
        
        # Ensure we have the requested number of paraphrases
        if len(paraphrases) < num_paraphrases:
            paraphrases.extend([user_msg] * (num_paraphrases - len(paraphrases)))
        paraphrases = paraphrases[:num_paraphrases]
        
        # All queries to process (original + paraphrases)
        all_queries = [user_msg] + paraphrases
        
        # Step 2: Retrieve documents for each query
        query_results = []
        all_unique_docs = []
        seen_docs = set()
        
        for i, query in enumerate(all_queries):
            query_type = "Original Query" if i == 0 else f"Paraphrase {i}"
            
            # Get relevant documents
            relevant_text = self.get_relevant_text(query, nresults=nresults, sim_th=sim_th)
            
            # Get detailed results for display
            query_result = self.collection.query(query_texts=query, n_results=nresults)
            
            # Process retrieved documents
            retrieved_docs = []
            if query_result.get('documents') and query_result['documents'][0]:
                for j, doc in enumerate(query_result['documents'][0]):
                    distance = query_result['distances'][0][j]
                    similarity = round(1 - distance, 3)
                    
                    # Only include if above similarity threshold
                    if similarity >= sim_th:
                        doc_info = {
                            'document': doc,
                            'similarity': similarity,
                            'metadata': query_result['metadatas'][0][j] if query_result.get('metadatas') else {}
                        }
                        retrieved_docs.append(doc_info)
                        
                        # Add to unique documents collection
                        if doc not in seen_docs:
                            seen_docs.add(doc)
                            all_unique_docs.append(doc)
            
            query_results.append({
                'query': query,
                'query_type': query_type,
                'documents': retrieved_docs,
                'total_found': len(retrieved_docs)
            })
        
        # Step 3: Generate final response using all unique documents
        if not all_unique_docs:
            return "No relevant documents found. Try lowering the relevance threshold.", query_results, self._format_query_results(query_results)
        
        # Create comprehensive context
        context_text = "\n\n".join([f"Context {i+1}: {doc}" for i, doc in enumerate(all_unique_docs)])
        
        # Generate final response
        enhanced_prompt = f"""Based on the following context information gathered from multiple search approaches, please provide a comprehensive answer to the question.

Context (from {len(all_unique_docs)} unique sources):
{context_text}

Original Question: {user_msg}

Please provide a detailed answer based on all the context provided:"""
        
        final_response = self.llm_generate(enhanced_prompt, top_k=top_k, top_p=top_p, temp=temp)
        
        # Format results for display
        formatted_results = self._format_query_results(query_results)
        
        return final_response, query_results, formatted_results

    # New wrapper methods using QueryTransformations class
    def multi_query_rag(
        self, 
        user_msg: str, 
        sim_th: float, 
        nresults: int, 
        top_k: int, 
        top_p: float, 
        temp: float,
        num_paraphrases: int = 3
    ) -> tuple[str, list[dict], str]:
        """
        Multi Query RAG: Generate multiple paraphrases, retrieve for each, then generate response
        Returns: (final_response, query_results, formatted_results)
        """
        return QueryTransformations.multi_query_rag(
            self, user_msg, sim_th, nresults, top_k, top_p, temp, num_paraphrases
        )

    def _format_query_results(self, query_results: list[dict]) -> str:
        """Format query results for display in the interface"""
        formatted = []
        
        for result in query_results:
            query_section = f"**{result['query_type']}:**\n"
            query_section += f"*{result['query']}*\n\n"
            
            if result['documents']:
                query_section += f"**Retrieved Documents ({result['total_found']}):**\n"
                for i, doc in enumerate(result['documents'], 1):
                    query_section += f"**Similarity: {doc['similarity']}**\n"
                    query_section += f"{doc['document'][:200]}{'...' if len(doc['document']) > 200 else ''}\n\n"
            else:
                query_section += "*No documents retrieved above threshold.*\n"
            
            query_section += "---\n"
            formatted.append(query_section)
        
        return "\n".join(formatted)

    def rag_fusion(
        self, 
        user_msg: str, 
        sim_th: float, 
        nresults: int, 
        top_k: int, 
        top_p: float, 
        temp: float,
        num_paraphrases: int = 3,
        fusion_threshold: float = 0.5,
        fusion_method: str = "max_score"
    ) -> tuple[str, list[dict], str, str]:
        """
        RAG Fusion: Generate paraphrases, retrieve for each, apply weighted fusion, then generate response
        Returns: (final_response, query_results, formatted_results, filtering_info)
        
        fusion_method options:
        - "max_score": Use the highest individual similarity score
        - "weighted_avg": Use weighted average of all scores  
        - "boost_multiple": Boost documents found by multiple queries
        - "top_n_only": Take top N documents regardless of threshold
        """
        logger.debug(f"rag_fusion args: sim_th: {sim_th}, nresults: {nresults}, fusion_threshold: {fusion_threshold}, method: {fusion_method}")
        
        # Step 1: Generate paraphrases (same as Multi Query)
        paraphrase_prompt = f"""Please provide {num_paraphrases} different ways to ask the same question. Each version should:
- Ask for the same information as the original
- Use different words and sentence structure
- NOT add any assumptions, context, or specificity that wasn't in the original
- Keep the same level of generality as the original question

Original question: {user_msg}

Provide only the alternative questions, one per line:"""
        
        paraphrases_response = self.llm_generate(paraphrase_prompt, top_k=top_k, top_p=top_p, temp=0.7)
        
        # Parse paraphrases
        paraphrases = [line.strip() for line in paraphrases_response.split('\n') if line.strip()]
        
        # Ensure we have the requested number of paraphrases
        if len(paraphrases) < num_paraphrases:
            paraphrases.extend([user_msg] * (num_paraphrases - len(paraphrases)))
        paraphrases = paraphrases[:num_paraphrases]
        
        # All queries to process (original + paraphrases)
        all_queries = [user_msg] + paraphrases
        
        # Step 2: Retrieve documents for each query and collect all results
        query_results = []
        doc_scores = {}  # document -> list of (query_index, similarity_score)
        all_docs_info = {}  # document -> metadata info
        
        for i, query in enumerate(all_queries):
            query_type = "Original Query" if i == 0 else f"Paraphrase {i}"
            
            # Get detailed results for this query
            query_result = self.collection.query(query_texts=query, n_results=nresults)
            
            # Process retrieved documents
            retrieved_docs = []
            if query_result.get('documents') and query_result['documents'][0]:
                for j, doc in enumerate(query_result['documents'][0]):
                    distance = query_result['distances'][0][j]
                    similarity = round(1 - distance, 3)
                    
                    # Only include if above similarity threshold
                    if similarity >= sim_th:
                        doc_info = {
                            'document': doc,
                            'similarity': similarity,
                            'metadata': query_result['metadatas'][0][j] if query_result.get('metadatas') else {}
                        }
                        retrieved_docs.append(doc_info)
                        
                        # Collect for fusion scoring
                        if doc not in doc_scores:
                            doc_scores[doc] = []
                            all_docs_info[doc] = doc_info
                        doc_scores[doc].append((i, similarity))
            
            query_results.append({
                'query': query,
                'query_type': query_type,
                'documents': retrieved_docs,
                'total_found': len(retrieved_docs)
            })
        
        # Step 3: Calculate fusion scores using selected method
        fusion_scores = {}
        for doc, scores in doc_scores.items():
            individual_scores = [score for _, score in scores]
            num_queries_found = len(scores)
            
            if fusion_method == "max_score":
                # Use the highest individual score
                fusion_score = max(individual_scores)
            elif fusion_method == "weighted_avg":
                # Traditional weighted average
                fusion_score = sum(individual_scores) / num_queries_found
            elif fusion_method == "boost_multiple":
                # Boost documents found by multiple queries
                max_score = max(individual_scores)
                boost_factor = 1 + (num_queries_found - 1) * 0.1  # 10% boost per additional query
                fusion_score = min(max_score * boost_factor, 1.0)  # Cap at 1.0
            elif fusion_method == "top_n_only":
                # Use max score but will filter by top N later
                fusion_score = max(individual_scores)
            else:
                # Default to weighted average
                fusion_score = sum(individual_scores) / num_queries_found
            
            fusion_scores[doc] = {
                'score': round(fusion_score, 3),
                'query_scores': scores,
                'num_queries': num_queries_found,
                'max_individual': round(max(individual_scores), 3),
                'avg_individual': round(sum(individual_scores) / len(individual_scores), 3)
            }
        
        # Step 4: Filter and rank documents
        before_filtering_count = len(fusion_scores)
        
        if fusion_method == "top_n_only":
            # Just take top N documents regardless of threshold
            target_count = min(nresults * 2, len(fusion_scores))  # Take up to 2x nresults
            ranked_docs = sorted(fusion_scores.items(), key=lambda x: x[1]['score'], reverse=True)
            filtered_docs = dict(ranked_docs[:target_count])
            after_filtering_count = len(filtered_docs)
            removed_count = before_filtering_count - after_filtering_count
            ranked_docs = list(filtered_docs.items())
        else:
            # Apply threshold filtering
            filtered_docs = {doc: info for doc, info in fusion_scores.items() 
                            if info['score'] >= fusion_threshold}
            after_filtering_count = len(filtered_docs)
            removed_count = before_filtering_count - after_filtering_count
            
            # Sort by fusion score (descending)
            ranked_docs = sorted(filtered_docs.items(), key=lambda x: x[1]['score'], reverse=True)
        
        # Step 5: Generate final response using top-ranked documents
        if not ranked_docs:
            filtering_info = self._format_fusion_filtering_info(
                before_filtering_count, after_filtering_count, removed_count, 
                fusion_threshold, ranked_docs, all_queries, fusion_method
            )
            return "No documents met the fusion criteria. Try a different filtering method or lower threshold.", query_results, self._format_query_results(query_results), filtering_info
        
        # Get final documents for response generation
        final_docs = [doc for doc, _ in ranked_docs]
        
        # Create comprehensive context
        context_text = "\n\n".join([f"Context {i+1}: {doc}" for i, doc in enumerate(final_docs)])
        
        # Generate final response
        enhanced_prompt = f"""Based on the following context information gathered and ranked using fusion scoring, please provide a comprehensive answer to the question.

Context (from {len(final_docs)} top-ranked sources):
{context_text}

Original Question: {user_msg}

Please provide a detailed answer based on all the context provided:"""
        
        final_response = self.llm_generate(enhanced_prompt, top_k=top_k, top_p=top_p, temp=temp)
        
        # Format results for display
        formatted_results = self._format_query_results(query_results)
        filtering_info = self._format_fusion_filtering_info(
            before_filtering_count, after_filtering_count, removed_count,
            fusion_threshold, ranked_docs, all_queries, fusion_method
        )
        
        return final_response, query_results, formatted_results, filtering_info

    def _format_fusion_filtering_info(self, before_count, after_count, removed_count, 
                                    threshold, ranked_docs, all_queries, fusion_method):
        """Format fusion filtering information for display"""
        info = []
        
        info.append("## 📊 **Fusion Filtering Process**\n")
        info.append(f"**Fusion Method:** {fusion_method}")
        info.append(f"**Total documents before filtering:** {before_count}")
        info.append(f"**Documents after filtering:** {after_count}")
        info.append(f"**Documents removed:** {removed_count}")
        
        if fusion_method != "top_n_only":
            info.append(f"**Fusion threshold:** {threshold}")
        
        info.append("")
        
        if ranked_docs:
            info.append("### **Re-ranking Scores:**\n")
            for i, (doc, fusion_info) in enumerate(ranked_docs[:10], 1):  # Show top 10
                info.append(f"**Document {i}:**")
                info.append(f"*{doc[:100]}{'...' if len(doc) > 100 else ''}*")
                
                # Show individual query scores
                for query_idx, similarity in fusion_info['query_scores']:
                    query_type = "Original Query" if query_idx == 0 else f"Paraphrase {query_idx}"
                    info.append(f"- {query_type} similarity: {similarity}")
                
                # Show fusion scoring details
                info.append(f"- **Max individual score: {fusion_info['max_individual']}**")
                info.append(f"- **Avg individual score: {fusion_info['avg_individual']}**")
                info.append(f"- **Final fusion score: {fusion_info['score']}** (method: {fusion_method})")
                info.append(f"- Found by {fusion_info['num_queries']} queries")
                info.append("")
        
        if removed_count > 0 and fusion_method != "top_n_only":
            info.append(f"### **Filtering Summary:**")
            info.append(f"Removed {removed_count} documents with fusion scores below {threshold}")
        elif fusion_method == "top_n_only":
            info.append(f"### **Filtering Summary:**")
            info.append(f"Selected top {after_count} documents regardless of threshold")
        
        return "\n".join(info)

    def decomposition_rag(
        self, 
        user_msg: str, 
        sim_th: float, 
        nresults: int, 
        top_k: int, 
        top_p: float, 
        temp: float,
        max_subqueries: int = 4,
        decomposition_method: str = "least_to_most"
    ) -> tuple[str, list[dict], str, str]:
        """
        Decomposition RAG: Break down complex queries into simpler subproblems
        Returns: (final_response, query_results, formatted_results, decomposition_info)
        
        decomposition_method options:
        - "least_to_most": Sequential decomposition with dependencies
        - "independent": Parallel decomposition without dependencies
        """
        logger.debug(f"decomposition_rag args: sim_th: {sim_th}, nresults: {nresults}, method: {decomposition_method}")
        
        # Step 1: Decompose the complex query into subproblems
        if decomposition_method == "least_to_most":
            decompose_prompt = f"""Break down this complex question into a sequence of simpler subquestions that build upon each other. Each subquestion should:
- Be simpler than the original question
- Help answer a part of the overall question
- Build logically on the previous subquestions where appropriate
- Be answerable with available information

Complex question: {user_msg}

Provide {max_subqueries} subquestions in logical order, one per line:"""
        else:  # independent
            decompose_prompt = f"""Break down this complex question into independent subquestions that can be answered separately. Each subquestion should:
- Focus on one specific aspect of the original question
- Be answerable independently without relying on other subquestions
- Collectively cover all aspects of the original question

Complex question: {user_msg}

Provide {max_subqueries} independent subquestions, one per line:"""
        
        decomposition_response = self.llm_generate(decompose_prompt, top_k=top_k, top_p=top_p, temp=0.7)
        
        # Parse subqueries
        subqueries = [line.strip() for line in decomposition_response.split('\n') if line.strip()]
        
        # Ensure we have reasonable number of subqueries
        if len(subqueries) < 2:
            subqueries.extend([user_msg] * (2 - len(subqueries)))
        subqueries = subqueries[:max_subqueries]
        
        # Step 2: Process subqueries based on method
        query_results = []
        subquery_answers = []
        context_accumulator = ""
        
        for i, subquery in enumerate(subqueries):
            logger.info(f"Processing subquery {i+1}/{len(subqueries)}: {subquery}")
            
            # For least-to-most, include previous context
            if decomposition_method == "least_to_most" and context_accumulator:
                enhanced_subquery = f"Context from previous steps: {context_accumulator}\n\nCurrent question: {subquery}"
            else:
                enhanced_subquery = subquery
            
            # Retrieve documents for this subquery
            query_result = self.collection.query(query_texts=enhanced_subquery, n_results=nresults)
            
            # Process retrieved documents
            results = []
            if (query_result.get('documents') and query_result['documents'] and 
                len(query_result['documents']) > 0 and query_result['documents'][0]):
                
                for j, doc in enumerate(query_result['documents'][0]):
                    if (query_result.get('distances') and query_result['distances'] and 
                        len(query_result['distances']) > 0 and len(query_result['distances'][0]) > j):
                        
                        distance = query_result['distances'][0][j]
                        similarity = round(1 - distance, 3)
                        
                        # Only include if above similarity threshold
                        if similarity >= sim_th:
                            metadata = {}
                            if (query_result.get('metadatas') and query_result['metadatas'] and 
                                len(query_result['metadatas']) > 0 and len(query_result['metadatas'][0]) > j):
                                metadata = query_result['metadatas'][0][j]
                            
                            doc_info = {
                                'content': doc,
                                'similarity': similarity,
                                'metadata': metadata
                            }
                            results.append(doc_info)
            
            if results:
                # Store results for debugging
                query_results.append({
                    'query': subquery,
                    'enhanced_query': enhanced_subquery if enhanced_subquery != subquery else None,
                    'results': results,
                    'step': i + 1
                })
                
                # Create context for this subquery
                context_text = "\n\n".join([f"Source {j+1}: {result['content']}" 
                                          for j, result in enumerate(results)])
                
                # Generate answer for this subquery
                if decomposition_method == "least_to_most" and context_accumulator:
                    subquery_prompt = f"""Based on the following context and building on previous findings, answer the specific question.

Previous context: {context_accumulator}

Current context:
{context_text}

Question: {subquery}

Provide a focused answer:"""
                else:
                    subquery_prompt = f"""Based on the following context, answer the specific question.

Context:
{context_text}

Question: {subquery}

Provide a focused answer:"""
                
                subquery_answer = self.llm_generate(subquery_prompt, top_k=top_k, top_p=top_p, temp=temp)
                subquery_answers.append({
                    'question': subquery,
                    'answer': subquery_answer,
                    'step': i + 1
                })
                
                # For least-to-most, accumulate context
                if decomposition_method == "least_to_most":
                    context_accumulator += f"\n\nStep {i+1} - Q: {subquery}\nA: {subquery_answer}"
            else:
                # No documents found for this subquery
                query_results.append({
                    'query': subquery,
                    'enhanced_query': None,
                    'results': [],
                    'step': i + 1
                })
                subquery_answers.append({
                    'question': subquery,
                    'answer': "No relevant information found for this subquery.",
                    'step': i + 1
                })
        
        # Step 3: Synthesize final answer from subquery answers
        if subquery_answers:
            synthesis_prompt = f"""Based on the answers to the following subquestions, provide a comprehensive answer to the original complex question.

Original question: {user_msg}

Subquestion answers:
"""
            for answer_data in subquery_answers:
                synthesis_prompt += f"\nStep {answer_data['step']} - Q: {answer_data['question']}\nA: {answer_data['answer']}\n"
            
            synthesis_prompt += f"""\nNow synthesize these answers to provide a complete response to: {user_msg}"""
            
            final_response = self.llm_generate(synthesis_prompt, top_k=top_k, top_p=top_p, temp=temp)
        else:
            final_response = "Unable to decompose and answer the query due to lack of relevant information."
        
        # Format results for display
        formatted_results = self._format_decomposition_results(query_results, subquery_answers)
        
        # Create decomposition info
        decomposition_info = self._format_decomposition_info(
            user_msg, subqueries, subquery_answers, decomposition_method
        )
        
        return final_response, query_results, formatted_results, decomposition_info

    def _format_decomposition_results(self, query_results, subquery_answers):
        """Format decomposition results for display"""
        formatted = []
        
        for i, (query_data, answer_data) in enumerate(zip(query_results, subquery_answers)):
            formatted.append(f"## Step {i+1}: {query_data['query']}")
            
            if query_data['enhanced_query']:
                formatted.append(f"**Enhanced query:** {query_data['enhanced_query']}")
            
            if query_data['results']:
                formatted.append(f"**Found {len(query_data['results'])} relevant documents**")
                for j, result in enumerate(query_data['results'][:3]):  # Show top 3
                    formatted.append(f"**Document {j+1}:** {result['content'][:200]}...")
                    formatted.append(f"*Similarity: {result['similarity']:.3f}*")
            else:
                formatted.append("**No relevant documents found**")
            
            formatted.append(f"**Answer:** {answer_data['answer']}")
            formatted.append("")
        
        return "\n".join(formatted)

    def _format_decomposition_info(self, original_query, subqueries, answers, method):
        """Format decomposition information for display"""
        info = []
        
        info.append("## 🔍 **Decomposition Process**\n")
        info.append(f"**Method:** {method}")
        info.append(f"**Original Query:** {original_query}")
        info.append(f"**Number of Subqueries:** {len(subqueries)}")
        info.append("")
        
        info.append("### **Decomposition Steps:**\n")
        for i, (subquery, answer_data) in enumerate(zip(subqueries, answers), 1):
            info.append(f"**Step {i}:**")
            info.append(f"*Question:* {subquery}")
            info.append(f"*Answer:* {answer_data['answer'][:150]}{'...' if len(answer_data['answer']) > 150 else ''}")
            info.append("")
        
        if method == "least_to_most":
            info.append("### **Sequential Processing:**")
            info.append("Each step builds upon the previous answers, creating a chain of reasoning.")
        else:
            info.append("### **Independent Processing:**")
            info.append("Each subquery was answered independently, then synthesized.")
        
        return "\n".join(info)

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
        "Vector Store Initialization Setup with automatic synchronization"

        if self._check_initdb_conditions():
            logger.debug("Init DB conditions are met")
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
                    # Auto-sync existing collection with current data directory
                    logger.info("Checking for changes in data directory...")
                    self.sync_collection_with_data()
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
