import sys
import platform
import subprocess
import os
import time
import re
import logging
from pathlib import Path
from datetime import datetime
from functools import wraps
from typing import List, Dict, Any, Optional
import numpy as np
import weaviate
import weaviate.classes as wvc
from weaviate.util import generate_uuid5
from pypdf import PdfReader

from rag_models import LocalHuggingFaceEmbeddings, LocalHuggingFaceChatModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Setup logging
logging.basicConfig(
    filename='rag_system.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filemode='a'
)


# ==========================================
#        CONFIGURATION
# ==========================================

# Use local models
EMBEDDING_SOURCE = "local"
LLM_SOURCE = "local"

# Local models
# For embeddings: use open model (no authentication required)
LOCAL_EMBEDDING_MODEL_NAME = "google/embeddinggemma-300m"  # 768 dimensions

# For LLM: use lightweight model for CPU
LOCAL_LLM_MODEL_NAME = "google/gemma-3-1b-it"

# Weaviate configuration
WEAVIATE_CONTAINER_NAME = "android-interview-rag"
WEAVIATE_IMAGE = "semitechnologies/weaviate:1.33.7"
WEAVIATE_HTTP_PORT = 8081
WEAVIATE_GRPC_PORT = 50052

COLLECTION_NAME = "AndroidInterview"

# Query validation settings
MAX_QUERY_LENGTH = 500
DANGEROUS_PATTERNS = ['<script', 'javascript:', 'onerror=', 'onload=', 'eval(']


# ==========================================
#        UTILITY FUNCTIONS
# ==========================================

def validate_query(query: str, max_length: int = MAX_QUERY_LENGTH) -> str:
    """
    Validate and sanitize user query.
    
    Args:
        query: User query string
        max_length: Maximum allowed query length
    
    Returns:
        Sanitized query string
    
    Raises:
        ValueError: If query is invalid or unsafe
    """
    if not query or not query.strip():
        raise ValueError("Query cannot be empty")
    
    # Check length
    if len(query) > max_length:
        logging.warning(f"Query truncated from {len(query)} to {max_length} characters")
        query = query[:max_length]
    
    # Basic injection protection
    query_lower = query.lower()
    for pattern in DANGEROUS_PATTERNS:
        if pattern in query_lower:
            logging.warning(f"Potentially unsafe query detected: {pattern}")
            raise ValueError(f"Potentially unsafe query detected")
    
    # Clean up
    query = query.strip()
    return query


def extract_page_number(title: str) -> Optional[int]:
    """Extract page number from document title."""
    match = re.search(r'Page\s+(\d+)', title)
    if match:
        return int(match.group(1))
    return None


def extract_relevant_snippet(content: str, question: str, snippet_length: int = 200) -> str:
    """
    Extract most relevant snippet from content based on question keywords.
    
    Args:
        content: Full document content
        question: User question
        snippet_length: Maximum snippet length
    
    Returns:
        Relevant snippet string
    """
    # Extract keywords from question
    question_words = set(re.findall(r'\b\w+\b', question.lower()))
    
    # Find sentences with most keyword matches
    sentences = re.split(r'[.!?]\s+', content)
    sentence_scores = []
    
    for sentence in sentences:
        sentence_words = set(re.findall(r'\b\w+\b', sentence.lower()))
        matches = len(question_words.intersection(sentence_words))
        if matches > 0:
            sentence_scores.append((matches, sentence))
    
    if sentence_scores:
        # Get sentence with most matches
        sentence_scores.sort(reverse=True)
        best_sentence = sentence_scores[0][1]
        
        # Return snippet around best sentence
        start_idx = content.find(best_sentence)
        if start_idx >= 0:
            snippet = content[max(0, start_idx - 50):start_idx + snippet_length]
            return snippet.strip()
    
    # Fallback: return beginning of content
    return content[:snippet_length].strip() + "..." if len(content) > snippet_length else content


def calculate_confidence(sources: List[Dict]) -> float:
    """
    Calculate confidence score based on source distances.
    
    Args:
        sources: List of source documents with distance scores
    
    Returns:
        Confidence score between 0.0 and 1.0
    """
    if not sources:
        return 0.0
    
    # Average distance (lower is better)
    distances = [s.get('distance', 1.0) for s in sources]
    avg_distance = sum(distances) / len(distances)
    
    # Convert distance to confidence (0-1 scale)
    # Assuming cosine distance range 0-1, where 0 = perfect match
    confidence = max(0.0, min(1.0, 1.0 - avg_distance))
    
    # Boost confidence if we have multiple good sources
    if len(sources) >= 3 and avg_distance < 0.5:
        confidence = min(1.0, confidence * 1.1)
    
    return round(confidence, 4)


def retry_on_error(max_retries: int = 3, delay: float = 1.0):
    """
    Decorator for retrying functions on errors.
    
    Args:
        max_retries: Maximum number of retry attempts
        delay: Initial delay between retries (exponential backoff)
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except (ConnectionError, TimeoutError) as e:
                    last_exception = e
                    if attempt < max_retries - 1:
                        wait_time = delay * (2 ** attempt)  # Exponential backoff
                        logging.warning(f"Retry {attempt + 1}/{max_retries} after {wait_time}s: {e}")
                        time.sleep(wait_time)
                    else:
                        logging.error(f"Max retries reached: {e}")
                        raise
                except Exception as e:
                    # Don't retry on other exceptions
                    raise
            if last_exception:
                raise last_exception
        return wrapper
    return decorator


# ==========================================
#        SHELL COMMAND HELPERS
# ==========================================

def run_shell_command(command):
    """Universal function to run a shell command."""
    system = platform.system()
    USE_WSL = system == "Windows"
    
    if USE_WSL:
        result = subprocess.run(
            ["wsl", "-e", "bash", "-l", "-c", command],
            capture_output=True, text=True, encoding="utf-8", errors="replace"
        )
    else:
        result = subprocess.run(
            command, shell=True, capture_output=True, text=True, encoding="utf-8", errors="replace"
        )
    return {
        "returncode": result.returncode,
        "stdout": result.stdout.strip(),
        "stderr": result.stderr.strip(),
        "success": result.returncode == 0
    }


# ==========================================
#        DATA LOADING
# ==========================================

def load_knowledge_base(pdf_path: str = None, chunk_size: int = 1000, chunk_overlap: int = 200):
    """
    Load knowledge base from PDF file using pypdf.
    
    Args:
        pdf_path: Path to PDF file. If None, uses default coroutines.pdf in module4 directory.
        chunk_size: Maximum size of text chunks in characters (default: 1000).
        chunk_overlap: Number of characters to overlap between chunks (default: 200).
    
    Returns:
        List of dictionaries with keys: title, content, tag
    """
    # Determine PDF path
    if pdf_path is None:
        # Get the directory where this script is located
        script_dir = Path(__file__).parent
        pdf_path = script_dir / "coroutines.pdf"
    else:
        pdf_path = Path(pdf_path)
    
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")
    
    print(f"📄 Loading PDF from: {pdf_path}")
    
    # Read PDF and extract text
    try:
        reader = PdfReader(str(pdf_path))
        total_pages = len(reader.pages)
        print(f"   Found {total_pages} pages in PDF")
        
        # Extract text from all pages
        full_text = ""
        page_texts = []
        
        for page_num, page in enumerate(reader.pages, start=1):
            try:
                page_text = page.extract_text()
                if page_text:
                    # Clean up text: remove excessive whitespace, normalize line breaks
                    page_text = re.sub(r'\s+', ' ', page_text)  # Replace multiple whitespace with single space
                    page_text = page_text.strip()
                    if page_text:
                        full_text += page_text + " "
                        page_texts.append({
                            "page": page_num,
                            "text": page_text
                        })
            except Exception as e:
                print(f"   ⚠️ Warning: Could not extract text from page {page_num}: {e}")
                continue
        
        if not full_text.strip():
            raise ValueError("No text could be extracted from PDF")
        
        print(f"   Extracted {len(full_text)} characters of text")
        
    except Exception as e:
        raise RuntimeError(f"Failed to read PDF file: {e}")
    
    # Split text into chunks
    documents_data = []
    
    # Strategy: Split by pages first, then further split large pages into smaller chunks
    for page_info in page_texts:
        page_text = page_info["text"]
        page_num = page_info["page"]
        
        # If page text is smaller than chunk_size, use it as-is
        if len(page_text) <= chunk_size:
            # Try to extract a title from the first line or use page number
            title = _extract_title_from_text(page_text, page_num)
            
            documents_data.append({
                "title": title,
                "content": page_text,
                "tag": "coroutines"
            })
        else:
            # Split large pages into smaller chunks with overlap
            chunks = _split_text_with_overlap(page_text, chunk_size, chunk_overlap)
            
            for chunk_idx, chunk_text in enumerate(chunks, start=1):
                title = _extract_title_from_text(chunk_text, page_num, chunk_idx)
                
                documents_data.append({
                    "title": title,
                    "content": chunk_text,
                    "tag": "coroutines"
                })
    
    print(f"✅ Loaded {len(documents_data)} documents from PDF.")
    print(f"   Topics: {set([doc['tag'] for doc in documents_data])}")
    
    return documents_data


def _extract_title_from_text(text: str, page_num: int, chunk_idx: int = None) -> str:
    """
    Extract a meaningful title from text chunk.
    
    Args:
        text: Text content
        page_num: Page number
        chunk_idx: Optional chunk index within page
    
    Returns:
        Title string
    """
    # Try to find a heading or first sentence
    lines = text.split('\n')
    first_line = lines[0].strip() if lines else ""
    
    # If first line looks like a heading (short, capitalized, no punctuation at end)
    if first_line and len(first_line) < 100 and not first_line.endswith('.'):
        # Clean up the title
        title = first_line.strip()
        # Remove excessive capitalization
        if title.isupper() and len(title) < 50:
            title = title.title()
        return f"Page {page_num}" + (f" - Chunk {chunk_idx}" if chunk_idx else "") + f": {title}"
    
    # Otherwise, use first meaningful sentence or excerpt
    sentences = re.split(r'[.!?]\s+', text)
    if sentences and sentences[0]:
        first_sentence = sentences[0].strip()
        if len(first_sentence) > 10 and len(first_sentence) < 150:
            return f"Page {page_num}" + (f" - Chunk {chunk_idx}" if chunk_idx else "") + f": {first_sentence[:100]}"
    
    # Fallback to page number
    return f"Page {page_num}" + (f" - Chunk {chunk_idx}" if chunk_idx else "")


def _split_text_with_overlap(text: str, chunk_size: int, overlap: int) -> list:
    """
    Split text into chunks with overlap between consecutive chunks.
    
    Args:
        text: Text to split
        chunk_size: Maximum size of each chunk
        overlap: Number of characters to overlap between chunks
    
    Returns:
        List of text chunks
    """
    if len(text) <= chunk_size:
        return [text]
    
    chunks = []
    start = 0
    
    while start < len(text):
        end = start + chunk_size
        
        # Try to break at sentence boundary if possible
        if end < len(text):
            # Look for sentence endings near the chunk boundary
            boundary_search = text[max(start, end - 100):end + 50]
            sentence_end = max(
                boundary_search.rfind('. '),
                boundary_search.rfind('.\n'),
                boundary_search.rfind('! '),
                boundary_search.rfind('? ')
            )
            
            if sentence_end > 0:
                end = start + (end - start - 100) + sentence_end + 1
        
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        
        # Move start position with overlap
        start = end - overlap
        if start >= len(text):
            break
    
    return chunks


# ==========================================
#        WEAVIATE SETUP
# ==========================================

def start_weaviate():
    """Start Weaviate container."""
    # Stop old container if exists
    print(f"--- Stopping and removing any existing container named '{WEAVIATE_CONTAINER_NAME}' ---")
    stop_command = f"docker stop {WEAVIATE_CONTAINER_NAME} 2>/dev/null; docker rm {WEAVIATE_CONTAINER_NAME} 2>/dev/null"
    run_shell_command(stop_command)
    print("Cleanup complete.")

    # Start new Weaviate container
    print(f"\n--- Starting Weaviate container '{WEAVIATE_CONTAINER_NAME}' ---")
    run_command = (
        f"docker run -d "
        f"--name {WEAVIATE_CONTAINER_NAME} "
        f"-p {WEAVIATE_HTTP_PORT}:8080 "
        f"-p {WEAVIATE_GRPC_PORT}:50051 "
        f"-e AUTHENTICATION_ANONYMOUS_ACCESS_ENABLED=true "
        f"-e PERSISTENCE_DATA_PATH=/var/lib/weaviate "
        f"-e DEFAULT_VECTORIZER_MODULE=none "
        f"-e ENABLE_MODULES='' "
        f"-e CLUSTER_HOSTNAME=node1 "
        f"{WEAVIATE_IMAGE}"
    )

    result = run_shell_command(run_command)

    if result["success"]:
        print("✅ Weaviate container started successfully.")
        print("Waiting a few seconds for the service to initialize...")
        time.sleep(10)
    else:
        print("❌ Failed to start Weaviate container.")
        print(f"Stderr: {result['stderr']}")
        raise RuntimeError("Failed to start Weaviate container")


# ==========================================
#        EMBEDDINGS AND VECTOR STORE
# ==========================================

def setup_rag_system(documents_data):
    """Setup models, generate embeddings, and populate Weaviate."""
    # --- Initialize models ---
    print("--- 1. Setting up embeddings model ---")
    try:
        embeddings_model = LocalHuggingFaceEmbeddings(LOCAL_EMBEDDING_MODEL_NAME)
        print("✅ Initialized.")
    except Exception as e:
        print(f"❌ Failed to initialize: {e}")
        raise

    # --- Generate embeddings ---
    print("\n--- 2. Generating embeddings for all documents ---")
    contents_to_embed = [doc['content'] for doc in documents_data]
    vector_embeddings = embeddings_model.embed_documents(contents_to_embed)
    print(f"✅ Generated {len(vector_embeddings)} embeddings. Vector dimension: {len(vector_embeddings[0])}")

    # Add embeddings to documents
    for i, doc in enumerate(documents_data):
        doc['content_vector'] = vector_embeddings[i]

    # --- Connect to Weaviate ---
    print("\n--- 3. Connecting to Weaviate ---")
    weaviate_client = weaviate.connect_to_local(
        host="localhost",
        port=WEAVIATE_HTTP_PORT,
        grpc_port=WEAVIATE_GRPC_PORT
    )

    if weaviate_client.is_ready():
        print("✅ Successfully connected to Weaviate.")
    else:
        print("❌ Failed to connect to Weaviate.")
        weaviate_client.close()
        raise ConnectionError("Could not connect to Weaviate instance.")

    # --- Create collection ---
    print(f"\n--- 4. Creating Weaviate collection: '{COLLECTION_NAME}' ---")

    if weaviate_client.collections.exists(COLLECTION_NAME):
        weaviate_client.collections.delete(COLLECTION_NAME)
        print(f"Deleted existing collection '{COLLECTION_NAME}'.")

    rag_collection = weaviate_client.collections.create(
        name=COLLECTION_NAME,
        properties=[
            wvc.config.Property(
                name="title", 
                data_type=wvc.config.DataType.TEXT,
                index_filterable=True,
                index_searchable=True
            ),
            wvc.config.Property(
                name="content", 
                data_type=wvc.config.DataType.TEXT,
                index_filterable=True,
                index_searchable=True
            ),
            wvc.config.Property(name="tag", data_type=wvc.config.DataType.TEXT),
        ],
        vector_config=wvc.config.Configure.Vectors.self_provided(
            vector_index_config=wvc.config.Configure.VectorIndex.hnsw(
                distance_metric=wvc.config.VectorDistances.COSINE
            )
        )
    )
    print(f"✅ Collection '{COLLECTION_NAME}' created successfully.")

    # --- Ingest data into Weaviate ---
    print(f"\n--- 5. Ingesting {len(documents_data)} documents into Weaviate ---")
    with rag_collection.batch.dynamic() as batch:
        for doc in documents_data:
            properties = {
                "title": doc["title"],
                "content": doc["content"],
                "tag": doc["tag"]
            }
            batch.add_object(
                properties=properties,
                vector=doc["content_vector"],
                uuid=generate_uuid5(doc["title"])
            )

    print(f"✅ Data ingestion complete. Total objects in collection: {len(rag_collection)}")
    print("✅ RAG system is ready!")
    
    weaviate_client.close()


# ==========================================
#        QUERY EXPANSION
# ==========================================

def expand_query(question: str, chat_model=None) -> str:
    """
    Expand query using LLM to make it more descriptive for vector search.
    
    Args:
        question: Original user question
        chat_model: Optional chat model (will create if None)
    
    Returns:
        Expanded query string
    """
    if chat_model is None:
        chat_model = LocalHuggingFaceChatModel(LOCAL_LLM_MODEL_NAME)
    
    expansion_prompt = ChatPromptTemplate.from_template(
        "You are an expert in information retrieval. "
        "Please rephrase the following user query to be more descriptive and detailed, "
        "making it suitable for a vector database search about Kotlin Coroutines. "
        "Return only the rephrased query, without any additional text, headers, or explanations. "
        "\n\nOriginal Query: '{query}'\n\nRephrased Query:"
    )
    
    expansion_chain = expansion_prompt | chat_model | StrOutputParser()
    
    try:
        expanded = expansion_chain.invoke({"query": question})
        return expanded.strip()
    except Exception as e:
        print(f"⚠️ Query expansion failed: {e}, using original query")
        return question


# ==========================================
#        HYBRID SEARCH
# ==========================================

def hybrid_search(
    question: str,
    rag_collection,
    embeddings_model,
    top_k: int = 5,
    vector_weight: float = 0.7,
    keyword_weight: float = 0.3
) -> list:
    """
    Hybrid search combining vector and keyword (BM25) search.
    
    Args:
        question: User question
        rag_collection: Weaviate collection
        embeddings_model: Embeddings model for vector search
        top_k: Number of documents to return
        vector_weight: Weight for vector search results (default: 0.7)
        keyword_weight: Weight for keyword search results (default: 0.3)
    
    Returns:
        List of retrieved documents with combined scores
    """
    # Vector search
    query_embedding = embeddings_model.embed_query(question)
    vector_results = rag_collection.query.near_vector(
        near_vector=query_embedding,
        limit=top_k * 2,
        return_metadata=wvc.query.MetadataQuery(distance=True)
    )
    
    # Keyword search (BM25) - search in both title and content
    keyword_results = rag_collection.query.bm25(
        query=question,
        query_properties=["title", "content"],
        limit=top_k * 2,
        return_metadata=wvc.query.MetadataQuery(score=True)
    )
    
    # Combine and deduplicate by UUID
    combined = {}
    
    # Process vector results
    for obj in vector_results.objects:
        doc_id = str(obj.uuid)
        # Convert distance to score (1.0 - distance, since distance is 0-1 for cosine)
        vector_score = 1.0 - obj.metadata.distance
        combined[doc_id] = {
            'object': obj,
            'vector_score': vector_score,
            'keyword_score': 0.0,
            'title': obj.properties['title'],
            'content': obj.properties['content'],
            'distance': obj.metadata.distance
        }
    
    # Process keyword results
    keyword_scores_list = []
    for obj in keyword_results.objects:
        doc_id = str(obj.uuid)
        # Try to get BM25 score from metadata
        try:
            keyword_score = getattr(obj.metadata, 'score', 0.0)
        except:
            keyword_score = 0.0
        
        keyword_scores_list.append(keyword_score)
        
        if doc_id in combined:
            combined[doc_id]['keyword_score'] = keyword_score
        else:
            combined[doc_id] = {
                'object': obj,
                'vector_score': 0.0,
                'keyword_score': keyword_score,
                'title': obj.properties['title'],
                'content': obj.properties['content'],
                'distance': 1.0  # Default distance for keyword-only results
            }
    
    # Normalize scores to 0-1 range for fair combination
    if combined:
        # Normalize vector scores
        vector_scores = [doc['vector_score'] for doc in combined.values() if doc['vector_score'] > 0]
        if vector_scores:
            max_vector = max(vector_scores)
            if max_vector > 0:
                for doc in combined.values():
                    doc['vector_score'] = doc['vector_score'] / max_vector
        
        # Normalize keyword scores (BM25 scores can be large, normalize to 0-1)
        keyword_scores = [doc['keyword_score'] for doc in combined.values() if doc['keyword_score'] > 0]
        if keyword_scores:
            max_keyword = max(keyword_scores)
            min_keyword = min(keyword_scores)
            # Min-max normalization
            if max_keyword > min_keyword:
                for doc in combined.values():
                    if doc['keyword_score'] > 0:
                        doc['keyword_score'] = (doc['keyword_score'] - min_keyword) / (max_keyword - min_keyword)
                    else:
                        doc['keyword_score'] = 0.0
            elif max_keyword > 0:
                # All scores are the same, normalize to 1.0
                for doc in combined.values():
                    if doc['keyword_score'] > 0:
                        doc['keyword_score'] = 1.0
    
    # Calculate combined scores
    scored_docs = []
    for doc_id, data in combined.items():
        combined_score = vector_weight * data['vector_score'] + keyword_weight * data['keyword_score']
        scored_docs.append({
            'title': data['title'],
            'content': data['content'],
            'distance': data['distance'],
            'vector_score': data['vector_score'],
            'keyword_score': data['keyword_score'],
            'hybrid_score': combined_score,
            'uuid': doc_id
        })
    
    # Sort by combined score (higher is better)
    scored_docs.sort(key=lambda x: x['hybrid_score'], reverse=True)
    
    # Return top K
    return scored_docs[:top_k]


# ==========================================
#        RE-RANKING
# ==========================================

def rerank_documents(question: str, retrieved_docs: list, reranker=None) -> list:
    """
    Re-rank retrieved documents using cross-encoder model.
    
    Args:
        question: User question
        retrieved_docs: List of dicts with 'content' key
        reranker: Optional CrossEncoder model (will create if None)
    
    Returns:
        Re-ranked list of documents
    """
    if not retrieved_docs:
        return retrieved_docs
    
    try:
        if reranker is None:
            from sentence_transformers import CrossEncoder
            reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
        
        # Create pairs for cross-encoder
        pairs = [[question, doc['content']] for doc in retrieved_docs]
        
        # Get scores (convert to list to avoid numpy array issues)
        scores = reranker.predict(pairs)
        # Convert numpy array to list if needed
        if isinstance(scores, np.ndarray):
            scores = scores.tolist()
        
        # Sort by score (higher is better)
        reranked_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        reranked_docs = [retrieved_docs[i] for i in reranked_indices]
        
        # Update distances with reranking scores
        max_score = max(scores) if scores else 1.0
        for i, doc in enumerate(reranked_docs):
            score_value = float(scores[reranked_indices[i]])
            doc['rerank_score'] = score_value
            # Normalize score to distance-like metric (lower is better)
            doc['rerank_distance'] = 1.0 - (score_value / max_score) if max_score > 0 else 1.0
        
        return reranked_docs
    except Exception as e:
        print(f"⚠️ Re-ranking failed: {e}, using original order")
        return retrieved_docs


# ==========================================
#        RAG QUERY FUNCTION
# ==========================================

@retry_on_error(max_retries=3, delay=1.0)
def query_rag(question: str, top_k: int = 5, use_query_expansion: bool = False, use_reranking: bool = False, use_hybrid_search: bool = False):
    """
    Query the RAG system with a question.
    
    Args:
        question: User's question
        top_k: Number of documents to retrieve (default: 5)
        use_query_expansion: Whether to expand query using LLM (default: False)
        use_reranking: Whether to re-rank results using cross-encoder (default: False)
        use_hybrid_search: Whether to use hybrid search (vector + keyword) (default: False)
    
    Returns:
        Dictionary with answer, context, sources, confidence, and metadata
    """
    start_time = time.time()
    
    # Validate and sanitize query
    try:
        question = validate_query(question)
    except ValueError as e:
        logging.error(f"Query validation failed: {e}")
        return {
            "answer": f"Invalid query: {str(e)}. Please provide a valid question.",
            "context": "",
            "sources": [],
            "confidence": 0.0,
            "error": "validation_error",
            "metadata": {
                "processing_time_ms": 0,
                "timestamp": datetime.now().isoformat()
            }
        }
    
    logging.info(f"Query received: {question[:100]}...")
    
    print(f"\n{'='*60}")
    print(f"🔍 Querying RAG system")
    print(f"{'='*60}")
    print(f"Question: {question}")
    print()
    
    # Initialize models
    print("--- 1. Loading models ---")
    embeddings_model = LocalHuggingFaceEmbeddings(LOCAL_EMBEDDING_MODEL_NAME)
    chat_model = LocalHuggingFaceChatModel(LOCAL_LLM_MODEL_NAME)
    print("✅ Models loaded.")
    print()
    
    # Connect to Weaviate
    print("--- 2. Connecting to Weaviate ---")
    weaviate_client = weaviate.connect_to_local(
        host="localhost",
        port=WEAVIATE_HTTP_PORT,
        grpc_port=WEAVIATE_GRPC_PORT
    )
    
    if not weaviate_client.is_ready():
        weaviate_client.close()
        raise ConnectionError("Could not connect to Weaviate instance.")
    
    rag_collection = weaviate_client.collections.get(COLLECTION_NAME)
    print("✅ Connected to Weaviate.")
    print()
    
    # Expand query if enabled
    search_query = question
    if use_query_expansion:
        print("--- 3. Expanding query ---")
        search_query = expand_query(question, chat_model)
        print(f"   Original: {question}")
        print(f"   Expanded: {search_query}")
        print("✅ Query expanded.")
        print()
    
    # Generate embedding for the question (or expanded query)
    print(f"--- {'4' if not use_query_expansion else '4'}. Generating query embedding ---")
    query_embedding = embeddings_model.embed_query(search_query)
    print("✅ Query embedding generated.")
    print()
    
    # Search for similar documents
    retrieve_limit = top_k * 2 if use_reranking else top_k
    step_num = '5' if not use_query_expansion else '5'
    
    if use_hybrid_search:
        print(f"--- {step_num}. Hybrid search (vector + keyword) for top {retrieve_limit} documents ---")
        retrieved_docs = hybrid_search(
            question=search_query,
            rag_collection=rag_collection,
            embeddings_model=embeddings_model,
            top_k=retrieve_limit
        )
        
        if not retrieved_docs:
            weaviate_client.close()
            elapsed_time = (time.time() - start_time) * 1000
            logging.warning(f"No documents found for query: {question[:100]}")
            return {
                "answer": "I couldn't find relevant information in the knowledge base to answer this question. Please try rephrasing your question or using different keywords.",
                "context": "",
                "sources": [],
                "confidence": 0.0,
                "metadata": {
                    "processing_time_ms": int(elapsed_time),
                    "sources_count": 0,
                    "retrieval_method": "hybrid",
                    "model_version": LOCAL_LLM_MODEL_NAME,
                    "timestamp": datetime.now().isoformat()
                }
            }
        
        print(f"✅ Found {len(retrieved_docs)} relevant documents via hybrid search.")
        print()
    else:
        print(f"--- {step_num}. Vector search for top {retrieve_limit} similar documents ---")
        retrieved_objects = rag_collection.query.near_vector(
            near_vector=query_embedding,
            limit=retrieve_limit,
            return_metadata=wvc.query.MetadataQuery(distance=True)
        )
        
        if not retrieved_objects.objects:
            weaviate_client.close()
            elapsed_time = (time.time() - start_time) * 1000
            logging.warning(f"No documents found for query: {question[:100]}")
            return {
                "answer": "I couldn't find relevant information in the knowledge base to answer this question. Please try rephrasing your question or using different keywords.",
                "context": "",
                "sources": [],
                "confidence": 0.0,
                "metadata": {
                    "processing_time_ms": int(elapsed_time),
                    "sources_count": 0,
                    "retrieval_method": "vector",
                    "model_version": LOCAL_LLM_MODEL_NAME,
                    "timestamp": datetime.now().isoformat()
                }
            }
        
        print(f"✅ Found {len(retrieved_objects.objects)} relevant documents.")
        print()
        
        # Convert to list of dicts
        retrieved_docs = []
        for obj in retrieved_objects.objects:
            retrieved_docs.append({
                "title": obj.properties['title'],
                "content": obj.properties['content'],
                "distance": round(obj.metadata.distance, 4)
            })
    
    # Re-rank if enabled
    if use_reranking:
        print(f"--- {'6' if not use_query_expansion else '6'}. Re-ranking documents ---")
        retrieved_docs = rerank_documents(question, retrieved_docs)
        retrieved_docs = retrieved_docs[:top_k]  # Take top K after reranking
        print(f"✅ Documents re-ranked. Top {top_k} selected.")
        print()
    
    # Form context from retrieved documents
    context = "\n\n---\n\n".join([doc['content'] for doc in retrieved_docs])
    
    # Check for insufficient context
    if not context.strip():
        weaviate_client.close()
        elapsed_time = (time.time() - start_time) * 1000
        return {
            "answer": "I couldn't find relevant information in the knowledge base to answer this question.",
            "context": "",
            "sources": [],
            "confidence": 0.0,
            "metadata": {
                "processing_time_ms": int(elapsed_time),
                "sources_count": 0,
                "retrieval_method": "hybrid" if use_hybrid_search else "vector",
                "model_version": LOCAL_LLM_MODEL_NAME,
                "timestamp": datetime.now().isoformat()
            }
        }
    
    # Show retrieved sources with enhanced metadata
    step_num = '6' if not use_reranking else '7'
    if use_query_expansion:
        step_num = '6' if not use_reranking else '7'
    print(f"--- {step_num}. Retrieved sources ---")
    sources = []
    for i, doc in enumerate(retrieved_docs, 1):
        distance = doc.get('rerank_distance', doc.get('distance', 0.0))
        page_number = extract_page_number(doc['title'])
        relevance_score = 1.0 - distance  # Convert distance to relevance (higher is better)
        snippet = extract_relevant_snippet(doc['content'], question)
        
        source_info = {
            "source_number": i,
            "title": doc['title'],
            "content": doc['content'],
            "snippet": snippet,
            "distance": distance,
            "relevance_score": round(relevance_score, 4),
            "page_number": page_number
        }
        if 'rerank_score' in doc:
            source_info['rerank_score'] = doc['rerank_score']
        sources.append(source_info)
        
        print(f"  {i}. {doc['title']} (distance: {distance:.4f}, relevance: {relevance_score:.4f})")
        if page_number:
            print(f"     Page: {page_number}")
        if 'rerank_score' in doc:
            print(f"     Rerank score: {doc['rerank_score']:.4f}")
        print(f"     Snippet: {snippet[:150]}...")
    print()
    
    # Calculate overall confidence
    confidence = calculate_confidence(sources)
    
    # Warn if insufficient sources
    if len(sources) < 3:
        logging.warning(f"Only {len(sources)} sources retrieved for query: {question[:100]}")
    
    # Create sources list for prompt
    sources_list = "\n".join([
        f"Source {i+1}: {s['title']} (Page {s['page_number'] if s['page_number'] else 'N/A'})"
        for i, s in enumerate(sources)
    ])
    
    # Create LangChain prompt with role and explicit source citation
    print("--- 6. Creating prompt with role ---")
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
        "Available sources:\n{sources_list}\n\n"
        "Question: {question}\n\n"
        "Your answer (with source citations):"
    )
    print("✅ Prompt created.")
    print()
    
    # Create LangChain chain
    print("--- 7. Creating LangChain chain ---")
    answer_chain = prompt_template | chat_model | StrOutputParser()
    print("✅ Chain created.")
    print()
    
    # Generate answer
    print("--- 8. Generating answer ---")
    try:
        answer = answer_chain.invoke({
            "context": context,
            "question": question,
            "sources_list": sources_list
        })
        
        # Add warning if insufficient sources
        if len(sources) < 3:
            answer += "\n\n⚠️ Note: Limited information found in the knowledge base. This answer may be incomplete."
        
        print("✅ Answer generated.")
        print()
    except Exception as e:
        weaviate_client.close()
        elapsed_time = (time.time() - start_time) * 1000
        logging.error(f"Error generating answer: {e}", exc_info=True)
        return {
            "answer": "An error occurred while generating the answer. Please try rephrasing your question.",
            "context": context,
            "sources": sources,
            "confidence": calculate_confidence(sources),
            "error": "generation_error",
            "metadata": {
                "processing_time_ms": int(elapsed_time),
                "sources_count": len(sources),
                "retrieval_method": "hybrid" if use_hybrid_search else "vector",
                "model_version": LOCAL_LLM_MODEL_NAME,
                "timestamp": datetime.now().isoformat()
            }
        }
    
    weaviate_client.close()
    
    # Calculate processing time
    elapsed_time = (time.time() - start_time) * 1000
    
    # Log successful query
    logging.info(f"Query completed in {elapsed_time:.2f}ms with {len(sources)} sources, confidence: {confidence:.4f}")
    
    return {
        "answer": answer,
        "context": context,
        "sources": sources,
        "confidence": confidence,
        "metadata": {
            "processing_time_ms": int(elapsed_time),
            "sources_count": len(sources),
            "retrieval_method": "hybrid" if use_hybrid_search else ("reranked" if use_reranking else "vector"),
            "query_expansion": use_query_expansion,
            "model_version": LOCAL_LLM_MODEL_NAME,
            "embedding_model": LOCAL_EMBEDDING_MODEL_NAME,
            "timestamp": datetime.now().isoformat()
        }
    }


# ==========================================
#        CLEANUP FUNCTION
# ==========================================

def cleanup_resources(stop_weaviate_container: bool = False):
    """
    Clean up resources: close Weaviate connections and optionally stop container.
    
    Args:
        stop_weaviate_container: If True, stops and removes the Weaviate Docker container
    """
    print("=" * 60)
    print("🧹 Cleaning up resources...")
    print("=" * 60)
    
    # Close any open Weaviate connections
    # Note: Weaviate connections are typically closed in functions, but we ensure cleanup here
    print("--- Closing Weaviate connections ---")
    # Connections are already closed in individual functions, but we log it
    print("✅ Weaviate connections handled.")
    print()
    
    # Optionally stop and remove Weaviate container
    if stop_weaviate_container:
        print(f"--- Stopping and removing Weaviate container '{WEAVIATE_CONTAINER_NAME}' ---")
        stop_command = f"docker stop {WEAVIATE_CONTAINER_NAME} 2>/dev/null; docker rm {WEAVIATE_CONTAINER_NAME} 2>/dev/null"
        result = run_shell_command(stop_command)
        
        if result["success"]:
            print(f"✅ Container '{WEAVIATE_CONTAINER_NAME}' stopped and removed.")
        else:
            print(f"⚠️ Container might have already been stopped or doesn't exist.")
            if result["stderr"]:
                print(f"   Note: {result['stderr']}")
        print()
    else:
        print(f"ℹ️  Weaviate container '{WEAVIATE_CONTAINER_NAME}' left running.")
        print(f"   To stop it manually, run: docker stop {WEAVIATE_CONTAINER_NAME}")
        print()
    
    # Clear any cached models (Python garbage collector will handle this)
    print("--- Releasing model resources ---")
    # Models are automatically garbage collected when out of scope
    print("✅ Model resources released.")
    print()
    
    print("=" * 60)
    print("✅ Cleanup complete!")
    print("=" * 60)


# ==========================================
#        MAIN FUNCTION
# ==========================================

def main():
    """Main function to set up the Android Interview RAG system."""
    print("=" * 60)
    print("Android Interview Preparation RAG System")
    print("=" * 60)
    print(f"Configuration:")
    print(f"   Embeddings: {EMBEDDING_SOURCE} ({LOCAL_EMBEDDING_MODEL_NAME})")
    print(f"   LLM: {LLM_SOURCE} ({LOCAL_LLM_MODEL_NAME})")
    print("=" * 60)
    print()

    # Load knowledge base
    documents_data = load_knowledge_base()
    print()

    # Start Weaviate
    start_weaviate()
    print()

    # Setup RAG system
    setup_rag_system(documents_data)
    print()

    print("=" * 60)
    print("✅ Setup complete! RAG system is ready to use.")
    print("=" * 60)
    print()
    
    # Example query to demonstrate RAG usage
    print("=" * 60)
    print("📝 Example RAG Query")
    print("=" * 60)
    example_question = "What are coroutines and how do they differ from threads?"
    
    try:
        result = query_rag(example_question, top_k=5)
        
        print("=" * 60)
        print("💬 Generated Answer:")
        print("=" * 60)
        print(result["answer"])
        print()
        
        if "confidence" in result:
            print(f"Confidence Score: {result['confidence']:.4f}")
            print()
        
        print("=" * 60)
        print("📚 Retrieved Sources:")
        print("=" * 60)
        for source in result["sources"]:
            source_num = source.get("source_number", "?")
            page_info = f" (Page {source['page_number']})" if source.get('page_number') else ""
            print(f"{source_num}. {source['title']}{page_info}")
            print(f"   Distance: {source['distance']:.4f}, Relevance: {source.get('relevance_score', 0.0):.4f}")
            if 'snippet' in source:
                print(f"   Snippet: {source['snippet'][:150]}...")
            print()
        
        if "metadata" in result:
            print("=" * 60)
            print("📊 Metadata:")
            print("=" * 60)
            meta = result["metadata"]
            print(f"Processing Time: {meta.get('processing_time_ms', 0)}ms")
            print(f"Sources Count: {meta.get('sources_count', 0)}")
            print(f"Retrieval Method: {meta.get('retrieval_method', 'unknown')}")
            print(f"Model Version: {meta.get('model_version', 'unknown')}")
            print()
        
    except ValueError as e:
        print(f"❌ Validation Error: {e}")
        print("Please provide a valid question.")
    except ConnectionError as e:
        print(f"❌ Connection Error: {e}")
        print("Please make sure Weaviate is running.")
    except Exception as e:
        print(f"❌ Unexpected error during RAG query: {e}")
        logging.error(f"Unexpected error in main: {e}", exc_info=True)
        import traceback
        traceback.print_exc()
    
    print("=" * 60)
    print("✅ RAG system demonstration complete!")
    print("=" * 60)
    print()
    
    # Cleanup resources
    # Set stop_weaviate_container=True if you want to stop the Docker container
    # Set to False to leave it running for future queries
    # cleanup_resources(stop_weaviate_container=False)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user. Cleaning up...")
        cleanup_resources(stop_weaviate_container=False)
    except Exception as e:
        print(f"\n\n❌ Unexpected error occurred: {e}")
        import traceback
        traceback.print_exc()
        print("\n🧹 Cleaning up resources...")
        cleanup_resources(stop_weaviate_container=False)
        raise

