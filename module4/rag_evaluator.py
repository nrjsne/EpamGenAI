"""
RAG System Evaluator

This module provides automated evaluation of RAG system performance
using Retrieval Precision@K and Answer Faithfulness Score metrics.
"""

import json
import os
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime
import numpy as np

from rag_models import LocalHuggingFaceEmbeddings, LocalHuggingFaceChatModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import weaviate
import weaviate.classes as wvc

# Import configuration and functions from main RAG file
from android_interview_rag import (
    LOCAL_EMBEDDING_MODEL_NAME,
    LOCAL_LLM_MODEL_NAME,
    WEAVIATE_HTTP_PORT,
    WEAVIATE_GRPC_PORT,
    COLLECTION_NAME,
    hybrid_search
)


class RAGEvaluator:
    """Evaluator for RAG system performance metrics."""
    
    def __init__(self, use_query_expansion: bool = False, use_reranking: bool = False, use_hybrid_search: bool = False):
        """
        Initialize the RAG evaluator.
        
        Args:
            use_query_expansion: Whether to use query expansion (for improved system)
            use_reranking: Whether to use re-ranking (for improved system)
            use_hybrid_search: Whether to use hybrid search (vector + keyword) (for improved system)
        """
        self.use_query_expansion = use_query_expansion
        self.use_reranking = use_reranking
        self.use_hybrid_search = use_hybrid_search
        
        # Initialize models
        print("Loading models for evaluation...")
        self.embeddings_model = LocalHuggingFaceEmbeddings(LOCAL_EMBEDDING_MODEL_NAME)
        self.chat_model = LocalHuggingFaceChatModel(LOCAL_LLM_MODEL_NAME)
        
        # Initialize query expansion chain if needed
        if use_query_expansion:
            expansion_prompt = ChatPromptTemplate.from_template(
                "You are an expert in information retrieval. "
                "Please rephrase the following user query to be more descriptive and detailed, "
                "making it suitable for a vector database search about Kotlin Coroutines. "
                "Return only the rephrased query, without any additional text, headers, or explanations. "
                "\n\nOriginal Query: '{query}'\n\nRephrased Query:"
            )
            self.query_expansion_chain = expansion_prompt | self.chat_model | StrOutputParser()
        
        # Initialize re-ranker if needed
        if use_reranking:
            try:
                from sentence_transformers import CrossEncoder
                self.reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
                print("✅ Re-ranker loaded successfully.")
            except Exception as e:
                print(f"⚠️ Warning: Could not load re-ranker: {e}")
                self.reranker = None
                self.use_reranking = False
        else:
            self.reranker = None
        
        # Initialize faithfulness evaluator chain
        faithfulness_prompt = ChatPromptTemplate.from_template(
            "You are an expert evaluator. Your task is to evaluate how well an answer is based on the provided context.\n\n"
            "Context from knowledge base:\n{context}\n\n"
            "Question: {question}\n\n"
            "Answer: {answer}\n\n"
            "Evaluate the answer on a scale of 0.0 to 1.0 based on:\n"
            "1. How much of the answer is directly supported by the context (0.0 = none, 1.0 = fully supported)\n"
            "2. Whether the answer uses information from the context rather than general knowledge\n"
            "3. Whether the answer accurately represents what is in the context\n\n"
            "Respond with ONLY a single number between 0.0 and 1.0 (e.g., 0.75), no additional text."
        )
        self.faithfulness_chain = faithfulness_prompt | self.chat_model | StrOutputParser()
        
        print("✅ Evaluator initialized.")
    
    def connect_to_weaviate(self):
        """Connect to Weaviate instance."""
        weaviate_client = weaviate.connect_to_local(
            host="localhost",
            port=WEAVIATE_HTTP_PORT,
            grpc_port=WEAVIATE_GRPC_PORT
        )
        
        if not weaviate_client.is_ready():
            weaviate_client.close()
            raise ConnectionError("Could not connect to Weaviate instance.")
        
        rag_collection = weaviate_client.collections.get(COLLECTION_NAME)
        return weaviate_client, rag_collection
    
    def expand_query(self, question: str) -> str:
        """Expand query using LLM if query expansion is enabled."""
        if not self.use_query_expansion:
            return question
        
        try:
            expanded = self.query_expansion_chain.invoke({"query": question})
            return expanded.strip()
        except Exception as e:
            print(f"⚠️ Query expansion failed: {e}, using original query")
            return question
    
    def rerank_documents(self, question: str, retrieved_docs: List[Dict]) -> List[Dict]:
        """Re-rank documents using cross-encoder if re-ranking is enabled."""
        if not self.use_reranking or self.reranker is None or not retrieved_docs:
            return retrieved_docs
        
        try:
            # Create pairs for cross-encoder
            pairs = [[question, doc['content']] for doc in retrieved_docs]
            
            # Get scores (convert to list to avoid numpy array issues)
            scores = self.reranker.predict(pairs)
            # Convert numpy array to list if needed
            if isinstance(scores, np.ndarray):
                scores = scores.tolist()
            
            # Sort by score (higher is better)
            reranked_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
            reranked_docs = [retrieved_docs[i] for i in reranked_indices]
            
            # Update distances with reranking scores (normalized)
            max_score = max(scores) if scores else 1.0
            for i, doc in enumerate(reranked_docs):
                score_value = float(scores[reranked_indices[i]])
                doc['rerank_score'] = score_value
                doc['rerank_distance'] = 1.0 - (score_value / max_score) if max_score > 0 else 1.0
            
            return reranked_docs
        except Exception as e:
            print(f"⚠️ Re-ranking failed: {e}, using original order")
            return retrieved_docs
    
    def retrieve_documents(self, question: str, top_k: int = 5) -> List[Dict]:
        """Retrieve documents from Weaviate."""
        weaviate_client, rag_collection = self.connect_to_weaviate()
        
        try:
            # Expand query if enabled
            search_query = self.expand_query(question)
            
            retrieve_limit = top_k * 2 if self.use_reranking else top_k
            
            if self.use_hybrid_search:
                # Use hybrid search (vector + keyword)
                retrieved_docs = hybrid_search(
                    question=search_query,
                    rag_collection=rag_collection,
                    embeddings_model=self.embeddings_model,
                    top_k=retrieve_limit
                )
                # Convert hybrid search format to standard format
                retrieved_docs = [
                    {
                        "title": doc['title'],
                        "content": doc['content'],
                        "distance": doc.get('distance', 1.0 - doc.get('hybrid_score', 0.0))
                    }
                    for doc in retrieved_docs
                ]
            else:
                # Use vector search only
                query_embedding = self.embeddings_model.embed_query(search_query)
                
                # Retrieve documents
                retrieved_objects = rag_collection.query.near_vector(
                    near_vector=query_embedding,
                    limit=retrieve_limit,
                    return_metadata=wvc.query.MetadataQuery(distance=True)
                )
                
                # Convert to list of dicts
                retrieved_docs = []
                for obj in retrieved_objects.objects:
                    retrieved_docs.append({
                        "title": obj.properties['title'],
                        "content": obj.properties['content'],
                        "distance": round(obj.metadata.distance, 4)
                    })
            
            # Re-rank if enabled
            if self.use_reranking:
                retrieved_docs = self.rerank_documents(question, retrieved_docs)
                retrieved_docs = retrieved_docs[:top_k]  # Take top K after reranking
            
            return retrieved_docs
        finally:
            weaviate_client.close()
    
    def evaluate_retrieval_precision(
        self, 
        question: str, 
        retrieved_docs: List[Dict],
        expected_keywords: List[str],
        expected_concepts: List[str],
        top_k: int = 5
    ) -> Dict[str, Any]:
        """
        Evaluate Retrieval Precision@K.
        
        A document is considered relevant if it contains at least one expected keyword
        or matches at least one expected concept.
        """
        if not retrieved_docs:
            return {
                "precision": 0.0,
                "relevant_count": 0,
                "total_retrieved": 0,
                "relevant_docs": []
            }
        
        relevant_docs = []
        relevant_count = 0
        
        for doc in retrieved_docs[:top_k]:
            content_lower = doc['content'].lower()
            title_lower = doc['title'].lower()
            combined_text = content_lower + " " + title_lower
            
            # Check for keyword matches
            keyword_matches = sum(1 for kw in expected_keywords if kw.lower() in combined_text)
            
            # Check for concept matches (more lenient - check if any word from concept is present)
            concept_matches = 0
            for concept in expected_concepts:
                concept_words = concept.lower().split()
                if len(concept_words) > 0:
                    # Check if at least 50% of concept words are present
                    words_found = sum(1 for word in concept_words if word in combined_text)
                    if words_found >= len(concept_words) * 0.5:
                        concept_matches += 1
            
            # Document is relevant if it has keyword matches or concept matches
            is_relevant = keyword_matches > 0 or concept_matches > 0
            
            if is_relevant:
                relevant_count += 1
                relevant_docs.append({
                    "title": doc['title'],
                    "keyword_matches": keyword_matches,
                    "concept_matches": concept_matches
                })
        
        precision = relevant_count / min(len(retrieved_docs), top_k) if retrieved_docs else 0.0
        
        return {
            "precision": round(precision, 4),
            "relevant_count": relevant_count,
            "total_retrieved": min(len(retrieved_docs), top_k),
            "relevant_docs": relevant_docs
        }
    
    def evaluate_answer_faithfulness(
        self,
        question: str,
        answer: str,
        context: str
    ) -> Dict[str, Any]:
        """
        Evaluate Answer Faithfulness Score using LLM-as-judge.
        
        Returns a score between 0.0 and 1.0 indicating how much the answer
        is based on the provided context.
        """
        try:
            response = self.faithfulness_chain.invoke({
                "question": question,
                "answer": answer,
                "context": context
            })
            
            # Extract numeric score from response
            import re
            score_match = re.search(r'(\d+\.?\d*)', response.strip())
            if score_match:
                score = float(score_match.group(1))
                # Clamp to [0, 1]
                score = max(0.0, min(1.0, score))
            else:
                # Fallback: try to parse as float directly
                try:
                    score = float(response.strip())
                    score = max(0.0, min(1.0, score))
                except:
                    score = 0.5  # Default if parsing fails
                    print(f"⚠️ Could not parse faithfulness score, using default: {response}")
            
            return {
                "faithfulness_score": round(score, 4),
                "raw_response": response.strip()
            }
        except Exception as e:
            print(f"⚠️ Faithfulness evaluation failed: {e}")
            return {
                "faithfulness_score": 0.0,
                "raw_response": f"Error: {str(e)}"
            }
    
    def generate_answer(self, question: str, context: str, sources: List[Dict] = None) -> str:
        """Generate answer using LLM with context."""
        # Create sources list if provided
        sources_list = ""
        if sources:
            sources_list = "\n".join([
                f"Source {i+1}: {s.get('title', 'Unknown')}"
                for i, s in enumerate(sources[:5])
            ])
        
        prompt_template = ChatPromptTemplate.from_template(
            "You are an experienced Android developer and technical interviewer helping someone prepare for a technical interview. "
            "Your task is to answer the user's question based on the provided context from the knowledge base about Kotlin Coroutines. "
            "Provide a clear, concise, and well-structured answer that demonstrates understanding of the concepts. "
            "\n\nIMPORTANT: For each fact or concept you mention, cite the source document number (e.g., [Source 1], [Source 2]). "
            "If you use information from multiple sources, cite all relevant sources. "
            "At the end of your answer, list all sources you used.\n\n"
            "If the context contains relevant information, use it to give a comprehensive answer. "
            "If the context does not contain enough information to fully answer the question, "
            "you can supplement with your general knowledge, but clearly indicate what comes from the context and what is general knowledge. "
            "If you use general knowledge, explicitly state 'Based on general knowledge' for that part.\n\n"
            "Context from knowledge base:\n{context}\n\n"
            "{sources_section}"
            "Question: {question}\n\n"
            "Your answer (with source citations):"
        )
        
        sources_section = f"Available sources:\n{sources_list}\n\n" if sources_list else ""
        
        answer_chain = prompt_template | self.chat_model | StrOutputParser()
        answer = answer_chain.invoke({
            "context": context,
            "question": question,
            "sources_section": sources_section
        })
        
        return answer
    
    def evaluate_single_question(
        self,
        test_item: Dict[str, Any],
        top_k: int = 5
    ) -> Dict[str, Any]:
        """Evaluate a single test question."""
        question = test_item["question"]
        expected_keywords = test_item.get("expected_keywords", [])
        expected_concepts = test_item.get("expected_concepts", [])
        
        print(f"\n{'='*60}")
        print(f"Evaluating Question {test_item['id']}: {question[:60]}...")
        print(f"{'='*60}")
        
        # Retrieve documents
        retrieved_docs = self.retrieve_documents(question, top_k=top_k)
        
        # Evaluate retrieval precision
        retrieval_metrics = self.evaluate_retrieval_precision(
            question=question,
            retrieved_docs=retrieved_docs,
            expected_keywords=expected_keywords,
            expected_concepts=expected_concepts,
            top_k=top_k
        )
        
        # Generate context and answer
        context = "\n\n---\n\n".join([doc['content'] for doc in retrieved_docs])
        answer = self.generate_answer(question, context, sources=retrieved_docs)
        
        # Evaluate answer faithfulness
        faithfulness_metrics = self.evaluate_answer_faithfulness(
            question=question,
            answer=answer,
            context=context
        )
        
        result = {
            "question_id": test_item["id"],
            "question": question,
            "topic": test_item.get("topic", "unknown"),
            "retrieval_metrics": retrieval_metrics,
            "faithfulness_metrics": faithfulness_metrics,
            "retrieved_docs_count": len(retrieved_docs),
            "retrieved_docs": [{"title": doc["title"], "distance": doc.get("distance", 0.0)} for doc in retrieved_docs[:5]]
        }
        
        print(f"  Precision@K: {retrieval_metrics['precision']:.4f}")
        print(f"  Faithfulness: {faithfulness_metrics['faithfulness_score']:.4f}")
        
        return result
    
    def run_full_evaluation(
        self,
        test_dataset_path: str = "test_questions.json",
        top_k: int = 5,
        output_file: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Run full evaluation suite on test dataset.
        
        Args:
            test_dataset_path: Path to test questions JSON file
            top_k: Number of documents to retrieve
            output_file: Optional path to save results JSON
        
        Returns:
            Dictionary with evaluation results
        """
        # Load test dataset
        script_dir = Path(__file__).parent
        dataset_path = script_dir / test_dataset_path
        
        if not dataset_path.exists():
            raise FileNotFoundError(f"Test dataset not found: {dataset_path}")
        
        with open(dataset_path, 'r', encoding='utf-8') as f:
            test_questions = json.load(f)
        
        print(f"\n{'='*60}")
        print(f"Starting RAG Evaluation")
        print(f"{'='*60}")
        print(f"Configuration:")
        print(f"  Query Expansion: {self.use_query_expansion}")
        print(f"  Re-ranking: {self.use_reranking}")
        print(f"  Hybrid Search: {self.use_hybrid_search}")
        print(f"  Top-K: {top_k}")
        print(f"  Test Questions: {len(test_questions)}")
        print(f"{'='*60}\n")
        
        # Evaluate each question
        results = []
        for test_item in test_questions:
            try:
                result = self.evaluate_single_question(test_item, top_k=top_k)
                results.append(result)
            except Exception as e:
                print(f"❌ Error evaluating question {test_item.get('id', 'unknown')}: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        # Calculate aggregate metrics
        if results:
            avg_precision = sum(r["retrieval_metrics"]["precision"] for r in results) / len(results)
            avg_faithfulness = sum(r["faithfulness_metrics"]["faithfulness_score"] for r in results) / len(results)
            
            # Calculate precision distribution
            precision_scores = [r["retrieval_metrics"]["precision"] for r in results]
            perfect_precision = sum(1 for p in precision_scores if p == 1.0)
            high_precision = sum(1 for p in precision_scores if p >= 0.8)
            
            # Calculate metrics for weak questions (precision < 0.8)
            weak_questions = [r for r in results if r["retrieval_metrics"]["precision"] < 0.8]
            weak_precision_scores = [r["retrieval_metrics"]["precision"] for r in weak_questions]
            avg_precision_weak = sum(weak_precision_scores) / len(weak_precision_scores) if weak_precision_scores else 0.0
            weak_questions_count = len(weak_questions)
            
            summary = {
                "evaluation_date": datetime.now().isoformat(),
                "configuration": {
                    "use_query_expansion": self.use_query_expansion,
                    "use_reranking": self.use_reranking,
                    "use_hybrid_search": self.use_hybrid_search,
                    "top_k": top_k,
                    "total_questions": len(test_questions),
                    "evaluated_questions": len(results)
                },
                "metrics": {
                    "average_precision_at_k": round(avg_precision, 4),
                    "average_faithfulness_score": round(avg_faithfulness, 4),
                    "perfect_precision_count": perfect_precision,
                    "high_precision_count": high_precision,
                    "perfect_precision_rate": round(perfect_precision / len(results), 4) if results else 0.0,
                    "high_precision_rate": round(high_precision / len(results), 4) if results else 0.0,
                    "weak_questions_count": weak_questions_count,
                    "average_precision_weak_questions": round(avg_precision_weak, 4) if weak_precision_scores else 0.0
                },
                "detailed_results": results
            }
        else:
            summary = {
                "evaluation_date": datetime.now().isoformat(),
                "error": "No results generated",
                "configuration": {
                    "use_query_expansion": self.use_query_expansion,
                    "use_reranking": self.use_reranking,
                    "use_hybrid_search": self.use_hybrid_search,
                    "top_k": top_k
                }
            }
        
        # Save results
        if output_file is None:
            config_suffix = ""
            if self.use_query_expansion:
                config_suffix += "_qe"
            if self.use_reranking:
                config_suffix += "_rr"
            if self.use_hybrid_search:
                config_suffix += "_hybrid"
            if not config_suffix:
                config_suffix = "_baseline"
            
            output_file = script_dir / f"evaluation_results{config_suffix}.json"
        
        output_path = Path(output_file)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        print(f"\n{'='*60}")
        print(f"Evaluation Complete")
        print(f"{'='*60}")
        if 'metrics' in summary:
            print(f"Average Precision@K: {summary['metrics']['average_precision_at_k']:.4f}")
            print(f"Average Faithfulness: {summary['metrics']['average_faithfulness_score']:.4f}")
            print(f"Perfect Precision Rate: {summary['metrics']['perfect_precision_rate']:.4f}")
            if 'average_precision_weak_questions' in summary['metrics']:
                print(f"Weak Questions (P<0.8): {summary['metrics']['weak_questions_count']}")
                print(f"Average Precision (Weak): {summary['metrics']['average_precision_weak_questions']:.4f}")
        else:
            print(f"⚠️ No metrics available: {summary.get('error', 'Unknown error')}")
        print(f"Results saved to: {output_path}")
        print(f"{'='*60}\n")
        
        return summary


if __name__ == "__main__":
    import sys
    
    # Parse command line arguments
    use_qe = "--query-expansion" in sys.argv or "-qe" in sys.argv
    use_rr = "--reranking" in sys.argv or "-rr" in sys.argv
    use_hybrid = "--hybrid" in sys.argv or "-hybrid" in sys.argv
    
    # Create evaluator
    evaluator = RAGEvaluator(use_query_expansion=use_qe, use_reranking=use_rr, use_hybrid_search=use_hybrid)
    
    # Run evaluation
    results = evaluator.run_full_evaluation()

