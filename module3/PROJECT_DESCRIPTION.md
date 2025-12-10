# Android Interview Preparation RAG System

## Video
https://tinyurl.com/cnrfy7x5

## Main Idea

The Android Interview Preparation RAG (Retrieval-Augmented Generation) System is an intelligent assistant designed to help developers prepare for Android developer technical interviews. The system combines semantic search capabilities with large language models to provide accurate, context-aware answers based on a curated knowledge base covering Android development, Kotlin programming, Coroutines, and Flow.

The system demonstrates the power of RAG by comparing answers generated with and without retrieved context, showing how augmenting LLM responses with domain-specific knowledge improves accuracy and relevance.

## Core Concepts

### 1. Retrieval-Augmented Generation (RAG)
RAG is a technique that enhances LLM responses by:
- **Retrieval Phase**: Searching a knowledge base for relevant documents using semantic similarity
- **Augmentation Phase**: Injecting retrieved context into the LLM prompt
- **Generation Phase**: Generating answers based on both the user's question and retrieved context

### 2. Semantic Search
The system uses vector embeddings to find semantically similar documents rather than exact keyword matches, enabling more intuitive and context-aware retrieval.

### 3. Comparison-Based Learning
The UI displays two answers side-by-side:
- **Without RAG**: Shows the LLM's general knowledge
- **With RAG**: Shows answers grounded in the specific knowledge base

This comparison helps users understand the value of RAG and see how domain-specific knowledge improves answer quality.

## Design Details

### Architecture

The system follows a modular, three-tier architecture:

```
┌─────────────────────────────────────────┐
│         Streamlit UI (Frontend)         │
│         (chat-ui.py)                    │
└──────────────┬──────────────────────────┘
               │
┌──────────────▼──────────────────────────┐
│      RAG Models Layer (rag_models.py)   │
│  - LocalHuggingFaceEmbeddings           │
│  - LocalHuggingFaceChatModel            │
└──────────────┬──────────────────────────┘
               │
┌──────────────▼──────────────────────────┐
│      Data & Infrastructure Layer         │
│  - Weaviate Vector Database             │
│  - Knowledge Base (104 documents)        │
└─────────────────────────────────────────┘
```

### Component Breakdown

#### 1. Backend Notebook (`androd-interview-rag.ipynb`)
- **Purpose**: Data ingestion and system setup
- **Responsibilities**:
  - Loading and preprocessing the knowledge base dataset
  - Generating embeddings for all documents
  - Creating Weaviate collection and indexing vectors
  - Setting up the vector database infrastructure

#### 2. Model Classes (`rag_models.py`)
- **Purpose**: Shared model wrappers for embeddings and LLM
- **Components**:
  - `LocalHuggingFaceEmbeddings`: Wraps sentence-transformers for generating embeddings
  - `LocalHuggingFaceChatModel`: Wraps transformers pipeline for text generation
- **Design Pattern**: Shared module used by both notebook and UI to avoid code duplication

#### 3. User Interface (`chat-ui.py`)
- **Purpose**: Interactive web interface for querying the RAG system
- **Features**:
  - Dual answer display (with/without RAG)
  - Source document visualization
  - Configurable retrieval parameters
  - Real-time answer generation

### Data Flow

1. **User Input**: User enters an interview question via Streamlit UI
2. **Query Embedding**: Question is converted to a vector using the embedding model
3. **Semantic Search**: Vector similarity search finds top-k relevant documents in Weaviate
4. **Context Formation**: Retrieved documents are combined into a context string
5. **Dual Generation**:
   - **No RAG**: LLM generates answer using only its training knowledge
   - **With RAG**: LLM generates answer using retrieved context + question
6. **Display**: Both answers are shown side-by-side for comparison

## Dataset Concept

### Structure

The knowledge base contains **104 documents** organized into 4 main categories:

1. **Android** (25+ terms): Core Android components and concepts
   - Activity, Fragment, Service, BroadcastReceiver, ContentProvider
   - ViewModel, LiveData, Room, WorkManager
   - Jetpack Compose, Navigation Component, Data Binding
   - Permissions, Resources, Manifest, etc.

2. **Kotlin** (25+ terms): Kotlin language features
   - Basic syntax: val, var, data class, sealed class
   - Null safety: safe call (?.), Elvis operator (?:)
   - Advanced features: extension functions, scope functions, lambdas
   - Type system: lateinit, lazy, reified types, etc.

3. **Coroutines** (30+ terms): Asynchronous programming
   - Core concepts: Coroutines, suspend functions, CoroutineScope
   - Dispatchers: Main, IO, Default
   - Builders: launch, async, withContext
   - Concurrency: structured concurrency, supervisorScope
   - Exception handling and cancellation

4. **Flow** (25+ terms): Reactive streams
   - Basic concepts: Flow, cold/hot streams, emit, collect
   - Operators: map, filter, transform, combine, debounce
   - State management: StateFlow, SharedFlow
   - Advanced: channelFlow, stateIn, shareIn

### Document Format

Each document follows a structured format:
```json
{
  "title": "Term Name",
  "description": "Detailed explanation of the concept...",
  "tag": "category (android/kotlin/coroutines/flow)"
}
```

### Content Strategy

- **Comprehensive Coverage**: Covers fundamental concepts interviewers commonly ask about
- **Technical Accuracy**: Descriptions include specific technical details, use cases, and relationships
- **Interview-Focused**: Content structured to help answer typical interview questions
- **Balanced Distribution**: Roughly equal coverage across all four categories

## System Technical Details

### Technology Stack

#### Models
- **Embedding Model**: `google/embeddinggemma-300m`
  - Dimensions: 768
  - Purpose: Convert text to vector representations
  - Usage: Both document indexing and query encoding

- **LLM Model**: `google/gemma-3-1b-it`
  - Size: 1 billion parameters
  - Type: Instruction-tuned model
  - Purpose: Generate answers from prompts
  - Device: CPU-optimized (device=-1)

#### Vector Database
- **Weaviate**: Open-source vector database
  - Version: 1.33.7
  - Index Type: HNSW (Hierarchical Navigable Small World)
  - Distance Metric: Cosine similarity
  - Ports: HTTP 8081, gRPC 50052
  - Configuration: Self-provided vectors (no auto-vectorization)

#### Framework
- **LangChain**: Orchestration framework
  - Components Used:
    - `ChatPromptTemplate`: Template-based prompt creation
    - `StrOutputParser`: Parse LLM outputs to strings
    - Chain composition: `prompt | model | parser`

#### UI Framework
- **Streamlit**: Web application framework
  - Features: Session state management, interactive widgets, real-time updates

### Key Technical Decisions

1. **Local Models**: All models run locally to ensure privacy and avoid API costs
2. **Self-Provided Vectors**: Weaviate configured to accept pre-computed embeddings for better control
3. **Dual Prompt Strategy**: Separate prompts for RAG and non-RAG scenarios to clearly demonstrate differences
4. **Modular Design**: Shared model classes prevent code duplication between notebook and UI

### Prompt Engineering

#### Without RAG Prompt
```
You are an experienced Android developer helping someone prepare for a technical interview.
Answer the following question based on your knowledge.
Question: {question}

Your answer:
```
- **Purpose**: Elicit general knowledge from LLM
- **Strategy**: Minimal instructions to allow natural response

#### With RAG Prompt
```
You are a factual assistant.
Your task is to answer the user's question based only on the provided context,
do not use common knowledge, do not correct mistakes in provided context.
Synthesize the information from the context into a concise, bullet-point summary.
Focus on specific details like names, numbers, and technical terms mentioned in the context.
If the context does not contain the information needed to answer the question,
you must state: 'The provided context does not contain the answer to this question.'

Context:
{context}

Question: {question}

Your answer:
```
- **Purpose**: Force LLM to ground answers in provided context
- **Strategy**: Explicit instructions to prevent hallucination and ensure factual accuracy

## Requirements

### Software Dependencies

#### Python Packages
- `streamlit==1.39.0`: Web UI framework
- `weaviate-client==4.18.3`: Vector database client
- `langchain==1.1.2`: RAG orchestration framework
- `langchain-community==0.4.1`: Community integrations
- `sentence-transformers==5.1.2`: Embedding model library
- `transformers`: HuggingFace transformers for LLM
- `torch`: PyTorch for model execution
- `accelerate==1.12.0`: Model acceleration utilities
- `huggingface-hub==0.36.0`: HuggingFace model hub access
- `python-dotenv==1.2.1`: Environment variable management

#### System Requirements
- **Docker**: Required for running Weaviate container
- **Python**: 3.13.5 (or compatible 3.x version)
- **Memory**: Minimum 4GB RAM (8GB+ recommended for model loading)
- **Storage**: ~2GB for models and dependencies
- **CPU**: Multi-core recommended (models run on CPU)

### Environment Setup

1. **HuggingFace Token** (optional but recommended):
   - Required for accessing gated models (Gemma models)
   - Set via environment variable: `HUGGINGFACE_API_TOKEN`
   - Or use `huggingface-cli login`

2. **Docker**:
   - Must be installed and running
   - Container automatically managed by notebook

### Setup Steps

1. **Install Dependencies**: Run first cell of notebook to install all packages
2. **Configure Models**: Ensure model names match in notebook and UI
3. **Run Notebook**: Execute all cells to:
   - Load dataset
   - Generate embeddings
   - Create Weaviate collection
   - Ingest documents
4. **Start UI**: Run `streamlit run module3/chat-ui.py`
5. **Verify**: Check that Weaviate is running on port 8081

## Limitations

### Model Limitations

1. **Model Size**: 
   - Using 1B parameter model limits answer quality compared to larger models
   - Answers may be less detailed or occasionally inaccurate
   - CPU-only execution results in slower inference

2. **Embedding Model**:
   - 768-dimensional embeddings may not capture all semantic nuances
   - Limited to English language understanding

### System Limitations

1. **Knowledge Base Scope**:
   - Limited to 104 documents covering Android/Kotlin/Coroutines/Flow
   - Does not include all possible interview topics
   - May not cover very recent Android/Kotlin updates

2. **Retrieval Quality**:
   - Semantic search may retrieve irrelevant documents for ambiguous queries
   - No query expansion or re-ranking implemented
   - Fixed top-k retrieval (no adaptive retrieval)

3. **Answer Quality**:
   - LLM may still hallucinate even with RAG context
   - No fact-checking or validation mechanism
   - Answers depend on quality of retrieved documents

4. **Performance**:
   - Local CPU execution is slow (several seconds per answer)
   - No caching of embeddings or answers
   - Sequential processing (no parallel answer generation)

5. **Scalability**:
   - Single-user interface (no multi-user support)
   - No persistence of conversation history
   - Limited to single Weaviate instance

6. **Error Handling**:
   - Basic error handling in place
   - No retry mechanisms for failed operations
   - Limited validation of user inputs

### Technical Constraints

1. **Docker Dependency**: System requires Docker for Weaviate, limiting portability
2. **Port Conflicts**: Fixed ports (8081, 50052) may conflict with other services
3. **Memory Usage**: Loading models consumes significant RAM
4. **Network**: Requires internet for initial model download (if not cached)

### Future Improvements

Potential enhancements to address limitations:
- GPU acceleration for faster inference
- Larger, more capable LLM models
- Query expansion and re-ranking
- Answer validation and fact-checking
- Conversation history and context
- Multi-user support with authentication
- Caching layer for common queries
- More comprehensive knowledge base
- Support for code examples and diagrams

## Conclusion

The Android Interview Preparation RAG System successfully demonstrates the application of RAG technology to a practical use case. By comparing answers with and without retrieved context, it provides valuable insights into how domain-specific knowledge enhances LLM responses. While the system has limitations, it serves as an effective learning tool and interview preparation assistant for Android developers.

