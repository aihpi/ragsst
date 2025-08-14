"""
Query Transformation Techniques for RAG Systems

This module contains implementations of various query transformation techniques
including Multi Query RAG, RAG Fusion, and Decomposition RAG.
"""

import logging
from typing import Tuple, List, Dict, Any

logger = logging.getLogger(__name__)


class QueryTransformations:
    """
    A collection of query transformation techniques for enhancing RAG systems.
    
    This class contains static methods that can be used by RAGTool to implement
    various query transformation strategies like Multi Query, RAG Fusion, and 
    Decomposition (Least-to-Most Prompting).
    """
    
    @staticmethod
    def multi_query_rag(
        rag_tool,
        user_msg: str, 
        sim_th: float, 
        nresults: int, 
        top_k: int, 
        top_p: float, 
        temp: float,
        num_paraphrases: int = 3
    ) -> Tuple[str, List[Dict], str]:
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
        
        paraphrases_response = rag_tool.llm_generate(paraphrase_prompt, top_k=top_k, top_p=top_p, temp=0.7)
        
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
            relevant_text = rag_tool.get_relevant_text(query, nresults=nresults, sim_th=sim_th)
            
            # Get detailed results for display
            query_result = rag_tool.collection.query(query_texts=query, n_results=nresults)
            
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
                        
                        # Add to unique collection if not seen before
                        if doc not in seen_docs:
                            all_unique_docs.append(doc)
                            seen_docs.add(doc)
            
            query_results.append({
                'query': query,
                'query_type': query_type,
                'retrieved_docs': retrieved_docs,
                'relevant_text': relevant_text
            })
        
        # Step 3: Generate final response using all unique documents
        if all_unique_docs:
            # Create comprehensive context
            context_text = "\n\n".join([f"Context {i+1}: {doc}" for i, doc in enumerate(all_unique_docs)])
            
            # Generate final response
            enhanced_prompt = f"""Based on the following context information gathered from multiple query perspectives, please provide a comprehensive answer to the question.

Context (from {len(all_unique_docs)} unique sources):
{context_text}

Original question: {user_msg}

Please provide a detailed answer:"""
            
            final_response = rag_tool.llm_generate(enhanced_prompt, top_k=top_k, top_p=top_p, temp=temp)
        else:
            final_response = "No relevant documents found for any of the query variations. Please try adjusting your query or lowering the similarity threshold."
        
        # Format results for display
        formatted_results = QueryTransformations._format_multi_query_results(query_results)
        
        return final_response, query_results, formatted_results

    @staticmethod
    def rag_fusion(
        rag_tool,
        user_msg: str, 
        sim_th: float, 
        nresults: int, 
        top_k: int, 
        top_p: float, 
        temp: float,
        num_paraphrases: int = 3,
        fusion_threshold: float = 0.5,
        fusion_method: str = "max_score"
    ) -> Tuple[str, List[Dict], str, str]:
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
        
        paraphrases_response = rag_tool.llm_generate(paraphrase_prompt, top_k=top_k, top_p=top_p, temp=0.7)
        
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
        
        for i, query in enumerate(all_queries):
            query_type = "Original Query" if i == 0 else f"Paraphrase {i}"
            
            # Get detailed results
            query_result = rag_tool.collection.query(query_texts=query, n_results=nresults)
            
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
                        
                        # Collect scores for fusion
                        if doc not in doc_scores:
                            doc_scores[doc] = []
                        doc_scores[doc].append((i, similarity))
            
            query_results.append({
                'query': query,
                'query_type': query_type,
                'retrieved_docs': retrieved_docs
            })
        
        # Step 3: Apply fusion scoring
        before_filtering_count = len(doc_scores)
        ranked_docs = QueryTransformations._apply_fusion_scoring(doc_scores, fusion_method, fusion_threshold, nresults)
        after_filtering_count = len(ranked_docs)
        removed_count = before_filtering_count - after_filtering_count
        
        # Step 4: Generate final response using top-ranked documents
        if not ranked_docs:
            filtering_info = QueryTransformations._format_fusion_filtering_info(
                before_filtering_count, after_filtering_count, removed_count, 
                fusion_threshold, ranked_docs, all_queries, fusion_method
            )
            return "No documents met the fusion criteria. Try a different filtering method or lower threshold.", query_results, QueryTransformations._format_query_results(query_results), filtering_info
        
        # Get final documents for response generation
        final_docs = [doc for doc, _ in ranked_docs]
        
        # Create comprehensive context
        context_text = "\n\n".join([f"Context {i+1}: {doc}" for i, doc in enumerate(final_docs)])
        
        # Generate final response
        enhanced_prompt = f"""Based on the following context information gathered and ranked using fusion scoring, please provide a comprehensive answer to the question.

Context (from {len(final_docs)} top-ranked sources):
{context_text}

Original question: {user_msg}

Please provide a detailed answer:"""
        
        final_response = rag_tool.llm_generate(enhanced_prompt, top_k=top_k, top_p=top_p, temp=temp)
        
        # Format results for display
        formatted_results = QueryTransformations._format_query_results(query_results)
        
        # Create filtering info
        filtering_info = QueryTransformations._format_fusion_filtering_info(
            before_filtering_count, after_filtering_count, removed_count,
            fusion_threshold, ranked_docs, all_queries, fusion_method
        )
        
        return final_response, query_results, formatted_results, filtering_info

    @staticmethod
    def decomposition_rag(
        rag_tool,
        user_msg: str, 
        sim_th: float, 
        nresults: int, 
        top_k: int, 
        top_p: float, 
        temp: float,
        max_subqueries: int = 4,
        decomposition_method: str = "least_to_most"
    ) -> Tuple[str, List[Dict], str, str]:
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
        
        decomposition_response = rag_tool.llm_generate(decompose_prompt, top_k=top_k, top_p=top_p, temp=0.7)
        
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
            query_result = rag_tool.collection.query(query_texts=enhanced_subquery, n_results=nresults)
            
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
                
                subquery_answer = rag_tool.llm_generate(subquery_prompt, top_k=top_k, top_p=top_p, temp=temp)
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
            
            final_response = rag_tool.llm_generate(synthesis_prompt, top_k=top_k, top_p=top_p, temp=temp)
        else:
            final_response = "Unable to decompose and answer the query due to lack of relevant information."
        
        # Format results for display
        formatted_results = QueryTransformations._format_decomposition_results(query_results, subquery_answers)
        
        # Create decomposition info
        decomposition_info = QueryTransformations._format_decomposition_info(
            user_msg, subqueries, subquery_answers, decomposition_method
        )
        
        return final_response, query_results, formatted_results, decomposition_info

    @staticmethod
    def hyde_rag(
        rag_tool,
        user_msg: str, 
        sim_th: float, 
        nresults: int, 
        top_k: int, 
        top_p: float, 
        temp: float,
        num_hypotheses: int = 1,
        hypothesis_length: str = "paragraph"
    ) -> Tuple[str, List[Dict], str, str]:
        """
        HyDE RAG: Generate hypothetical documents, use them for retrieval, then generate response
        Returns: (final_response, query_results, formatted_results, hyde_info)
        
        num_hypotheses: Number of hypothetical documents to generate
        hypothesis_length options:
        - "short": Brief answer (1-2 sentences)
        - "paragraph": Medium answer (1 paragraph)
        - "detailed": Longer answer (2-3 paragraphs)
        """
        logger.debug(f"hyde_rag args: sim_th: {sim_th}, nresults: {nresults}, num_hypotheses: {num_hypotheses}")
        
        # Step 1: Generate hypothetical document(s)
        if hypothesis_length == "short":
            length_instruction = "Write a brief, 1-2 sentence answer"
        elif hypothesis_length == "detailed":
            length_instruction = "Write a detailed, 2-3 paragraph answer"
        else:  # paragraph
            length_instruction = "Write a clear, one paragraph answer"
        
        hypotheses = []
        for i in range(num_hypotheses):
            if num_hypotheses == 1:
                hypothesis_prompt = f"""Imagine you are answering this question based on expert knowledge. {length_instruction} that directly addresses the query.

Question: {user_msg}

{length_instruction.replace('Write', 'Answer')}:"""
            else:
                hypothesis_prompt = f"""Imagine you are answering this question based on expert knowledge. {length_instruction} that directly addresses the query. Provide a variation that explores a different aspect or perspective.

Question: {user_msg}

{length_instruction.replace('Write', 'Answer')} (Variation {i+1}):"""
            
            hypothesis_response = rag_tool.llm_generate(hypothesis_prompt, top_k=top_k, top_p=top_p, temp=0.7)
            hypotheses.append(hypothesis_response.strip())
        
        # Step 2: Use hypothetical documents for retrieval
        query_results = []
        all_retrieved_docs = []
        doc_similarity_map = {}  # Track which hypothesis retrieved each doc
        
        for i, hypothesis in enumerate(hypotheses):
            logger.info(f"Using hypothesis {i+1}/{len(hypotheses)} for retrieval")
            
            # Use hypothesis as the search query instead of original question
            query_result = rag_tool.collection.query(query_texts=hypothesis, n_results=nresults)
            
            # Process retrieved documents
            retrieved_docs = []
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
                                'metadata': metadata,
                                'hypothesis_index': i + 1
                            }
                            retrieved_docs.append(doc_info)
                            
                            # Track document retrieval
                            if doc not in doc_similarity_map:
                                doc_similarity_map[doc] = []
                                all_retrieved_docs.append(doc)
                            doc_similarity_map[doc].append((i + 1, similarity))
            
            query_results.append({
                'hypothesis': hypothesis,
                'hypothesis_index': i + 1,
                'retrieved_docs': retrieved_docs,
                'total_found': len(retrieved_docs)
            })
        
        # Step 3: Generate final response using retrieved documents
        if all_retrieved_docs:
            # Create context from all unique retrieved documents
            context_text = "\n\n".join([f"Source {i+1}: {doc}" for i, doc in enumerate(all_retrieved_docs)])
            
            # Generate final response
            enhanced_prompt = f"""Based on the following context information retrieved using hypothetical document embeddings, please provide a comprehensive answer to the original question.

Context (from {len(all_retrieved_docs)} sources):
{context_text}

Original question: {user_msg}

Please provide a detailed and accurate answer based on the retrieved context:"""
            
            final_response = rag_tool.llm_generate(enhanced_prompt, top_k=top_k, top_p=top_p, temp=temp)
        else:
            final_response = "No relevant documents found using hypothetical document embeddings. Try adjusting the similarity threshold or generating different hypothetical documents."
        
        # Format results for display
        formatted_results = QueryTransformations._format_hyde_results(query_results, all_retrieved_docs, doc_similarity_map)
        
        # Create HyDE info
        hyde_info = QueryTransformations._format_hyde_info(
            user_msg, hypotheses, all_retrieved_docs, doc_similarity_map, hypothesis_length
        )
        
        return final_response, query_results, formatted_results, hyde_info

    # Helper methods for formatting and processing
    @staticmethod
    def _format_multi_query_results(query_results):
        """Format multi-query results for display"""
        formatted = []
        
        for result in query_results:
            formatted.append(f"## {result['query_type']}: {result['query']}")
            
            if result['retrieved_docs']:
                formatted.append(f"**Found {len(result['retrieved_docs'])} relevant documents**")
                for i, doc_info in enumerate(result['retrieved_docs'][:3]):  # Show top 3
                    formatted.append(f"**Document {i+1}:** {doc_info['document'][:200]}...")
                    formatted.append(f"*Similarity: {doc_info['similarity']:.3f}*")
            else:
                formatted.append("**No relevant documents found**")
            formatted.append("")
        
        return "\n".join(formatted)

    @staticmethod
    def _format_query_results(query_results):
        """Format query results for display"""
        formatted = []
        
        for result in query_results:
            formatted.append(f"## {result['query_type']}: {result['query']}")
            
            if result['retrieved_docs']:
                formatted.append(f"**Found {len(result['retrieved_docs'])} relevant documents**")
                for i, doc_info in enumerate(result['retrieved_docs'][:3]):  # Show top 3
                    formatted.append(f"**Document {i+1}:** {doc_info['document'][:200]}...")
                    formatted.append(f"*Similarity: {doc_info['similarity']:.3f}*")
            else:
                formatted.append("**No relevant documents found**")
            formatted.append("")
        
        return "\n".join(formatted)

    @staticmethod
    def _apply_fusion_scoring(doc_scores, fusion_method, fusion_threshold, nresults):
        """Apply fusion scoring to documents"""
        ranked_docs = []
        
        for doc, scores in doc_scores.items():
            # Calculate various fusion metrics
            individual_scores = [score for _, score in scores]
            max_individual = max(individual_scores)
            avg_individual = sum(individual_scores) / len(individual_scores)
            num_queries = len(scores)
            
            # Apply fusion method
            if fusion_method == "max_score":
                fusion_score = max_individual
            elif fusion_method == "weighted_avg":
                # Weight higher scores more heavily
                weights = [score for score in individual_scores]
                fusion_score = sum(s * w for s, w in zip(individual_scores, weights)) / sum(weights)
            elif fusion_method == "boost_multiple":
                # Use max score but add bonus for multiple query matches
                fusion_score = max_individual + (num_queries - 1) * 0.1
            elif fusion_method == "top_n_only":
                # Use max score but ignore threshold filtering
                fusion_score = max_individual
            else:
                fusion_score = max_individual
            
            # Store with detailed info for debugging
            fusion_info = {
                'score': fusion_score,
                'max_individual': max_individual,
                'avg_individual': avg_individual,
                'num_queries': num_queries,
                'query_scores': scores
            }
            
            # Apply threshold filtering (except for top_n_only)
            if fusion_method == "top_n_only" or fusion_score >= fusion_threshold:
                ranked_docs.append((doc, fusion_info))
        
        # Sort by fusion score (descending)
        ranked_docs.sort(key=lambda x: x[1]['score'], reverse=True)
        
        # For top_n_only, limit to nresults
        if fusion_method == "top_n_only":
            ranked_docs = ranked_docs[:nresults]
        
        return ranked_docs

    @staticmethod
    def _format_fusion_filtering_info(before_count, after_count, removed_count, 
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

    @staticmethod
    def _format_decomposition_results(query_results, subquery_answers):
        """Format decomposition results for display"""
        formatted = []
        
        for i, (query_data, answer_data) in enumerate(zip(query_results, subquery_answers)):
            formatted.append(f"## Step {i+1}: {query_data['query']}")
            
            if query_data.get('enhanced_query'):
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

    @staticmethod
    def _format_decomposition_info(original_query, subqueries, answers, method):
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

    @staticmethod
    def _format_hyde_results(query_results, all_retrieved_docs, doc_similarity_map):
        """Format HyDE results for display"""
        formatted = []
        
        for result in query_results:
            formatted.append(f"## Hypothesis {result['hypothesis_index']}")
            formatted.append(f"**Generated Hypothesis:**")
            formatted.append(f"*{result['hypothesis']}*")
            formatted.append("")
            
            if result['retrieved_docs']:
                formatted.append(f"**Found {len(result['retrieved_docs'])} relevant documents using this hypothesis**")
                for i, doc_info in enumerate(result['retrieved_docs'][:3]):  # Show top 3
                    formatted.append(f"**Document {i+1}:** {doc_info['content'][:200]}...")
                    formatted.append(f"*Similarity: {doc_info['similarity']:.3f}*")
            else:
                formatted.append("**No relevant documents found with this hypothesis**")
            formatted.append("")
        
        # Summary of all unique documents
        if all_retrieved_docs:
            formatted.append("## Summary of All Retrieved Documents")
            formatted.append(f"**Total unique documents retrieved:** {len(all_retrieved_docs)}")
            formatted.append("")
            
            for i, doc in enumerate(all_retrieved_docs[:5], 1):  # Show top 5
                formatted.append(f"**Document {i}:** {doc[:150]}...")
                # Show which hypotheses retrieved this document
                hypotheses_info = doc_similarity_map[doc]
                hyp_details = [f"Hypothesis {h_idx} (sim: {sim:.3f})" for h_idx, sim in hypotheses_info]
                formatted.append(f"*Retrieved by: {', '.join(hyp_details)}*")
                formatted.append("")
        
        return "\n".join(formatted)

    @staticmethod
    def _format_hyde_info(original_query, hypotheses, retrieved_docs, doc_similarity_map, hypothesis_length):
        """Format HyDE information for display"""
        info = []
        
        info.append("## 🔮 **HyDE Process**\n")
        info.append(f"**Original Query:** {original_query}")
        info.append(f"**Number of Hypotheses Generated:** {len(hypotheses)}")
        info.append(f"**Hypothesis Length:** {hypothesis_length}")
        info.append(f"**Total Unique Documents Retrieved:** {len(retrieved_docs)}")
        info.append("")
        
        info.append("### **Generated Hypotheses:**\n")
        for i, hypothesis in enumerate(hypotheses, 1):
            info.append(f"**Hypothesis {i}:**")
            info.append(f"*{hypothesis[:200]}{'...' if len(hypothesis) > 200 else ''}*")
            info.append("")
        
        info.append("### **Retrieval Strategy:**")
        info.append("Instead of using the original query for embedding similarity search, HyDE:")
        info.append("1. Generated hypothetical document(s) that would answer the query")
        info.append("2. Used these hypothetical documents as search queries")
        info.append("3. Retrieved documents similar to the hypothetical answers")
        info.append("4. Used retrieved documents to generate the final answer")
        info.append("")
        
        if retrieved_docs:
            info.append("### **Document Overlap:**")
            for doc in retrieved_docs[:3]:  # Show overlap for top 3 docs
                hypotheses_list = doc_similarity_map[doc]
                if len(hypotheses_list) > 1:
                    hyp_nums = [str(h_idx) for h_idx, _ in hypotheses_list]
                    info.append(f"- Document found by multiple hypotheses: {', '.join(hyp_nums)}")
                    info.append(f"  *{doc[:100]}{'...' if len(doc) > 100 else ''}*")
        
        return "\n".join(info)
