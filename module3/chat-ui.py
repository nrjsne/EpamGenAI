"""
UI for Android Interview RAG System

This UI allows users to interact with the RAG system through a web interface.
"""

import streamlit as st
import weaviate
import weaviate.classes as wvc
import os

# Import model classes from shared module
from rag_models import LocalHuggingFaceEmbeddings, LocalHuggingFaceChatModel

# Import LangChain components
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Configuration - must match the main RAG notebook
LOCAL_EMBEDDING_MODEL_NAME = "google/embeddinggemma-300m"
LOCAL_LLM_MODEL_NAME = "google/gemma-3-1b-it"
WEAVIATE_HTTP_PORT = 8081
WEAVIATE_GRPC_PORT = 50052
COLLECTION_NAME = "AndroidInterview"


# Page config
st.set_page_config(
    page_title="Android Interview RAG",
    page_icon="📱",
    layout="wide"
)

st.title("📱 Android Interview Preparation Assistant")
st.markdown("Ask questions about Android, Kotlin, Coroutines, and Flow to prepare for your interview!")

# Initialize session state
if "embeddings_model" not in st.session_state:
    with st.spinner("Loading embedding model..."):
        st.session_state.embeddings_model = LocalHuggingFaceEmbeddings(
            LOCAL_EMBEDDING_MODEL_NAME
        )

if "chat_model" not in st.session_state:
    with st.spinner("Loading LLM..."):
        hf_token = os.environ.get("HUGGINGFACE_API_TOKEN", None)
        try:
            st.session_state.chat_model = LocalHuggingFaceChatModel(
                LOCAL_LLM_MODEL_NAME
            )
        except Exception as e:
            st.error(f"Failed to load LLM: {e}")
            st.stop()

if "weaviate_client" not in st.session_state:
    with st.spinner("Connecting to Weaviate..."):
        try:
            st.session_state.weaviate_client = weaviate.connect_to_local(
                host="localhost",
                port=WEAVIATE_HTTP_PORT,
                grpc_port=WEAVIATE_GRPC_PORT
            )
            if st.session_state.weaviate_client.is_ready():
                st.session_state.rag_collection = st.session_state.weaviate_client.collections.get(COLLECTION_NAME)
            else:
                st.error("Failed to connect to Weaviate. Make sure it's running.")
                st.stop()
        except Exception as e:
            st.error(f"Failed to connect to Weaviate: {e}")
            st.error("💡 Make sure the main RAG notebook has been run and Weaviate is running.")
            st.stop()

# Initialize LangChain chains after models are loaded
if "answer_chain_no_rag" not in st.session_state and "chat_model" in st.session_state:
    # Prompt for answer generation WITHOUT RAG (no context)
    prompt_no_rag = ChatPromptTemplate.from_template(
        "You are an experienced Android developer helping someone prepare for a technical interview. "
        "Answer the following question based on your knowledge. "
        "Question: {question}\n\n"
        "Your answer:"
    )
    st.session_state.answer_chain_no_rag = prompt_no_rag | st.session_state.chat_model | StrOutputParser()

if "answer_chain_with_rag" not in st.session_state and "chat_model" in st.session_state:
    # Prompt for answer generation WITH RAG (with context from knowledge base)
    prompt_with_rag = ChatPromptTemplate.from_template(
        "You are a factual assistant. "
        "Your task is to answer the user's question based only on the provided context, "
        "do not use common knowledge, do not correct mistakes in provided context. "
        "Synthesize the information from the context into a concise, bullet-point summary. "
        "Focus on specific details like names, numbers, and technical terms mentioned in the context. "
        "If the context does not contain the information needed to answer the question, "
        "you must state: 'The provided context does not contain the answer to this question.' "
        "\n\nContext:\n{context}\n\nQuestion: {question}\n\n"
        "Your answer:"
    )
    st.session_state.answer_chain_with_rag = prompt_with_rag | st.session_state.chat_model | StrOutputParser()

# Sidebar for settings
with st.sidebar:
    st.header("⚙️ Settings")
    top_k = st.slider("Number of documents to retrieve",step=1,min_value=3, max_value=10, value=5)
    st.markdown("---")
    st.markdown("### 📚 About")
    st.markdown("This RAG system uses:")
    st.markdown("- **Embeddings:** google/embeddinggemma-300m")
    st.markdown("- **LLM:** google/gemma-3-1b-it")
    st.markdown("- **Vector DB:** Weaviate")
    st.markdown("- **Framework:** LangChain")
    st.markdown("---")
    st.markdown("### 💡 Tips")
    st.markdown("- Make sure the main RAG notebook has been run")
    st.markdown("- Weaviate should be running on port 8081")
    st.markdown("- Questions about Android, Kotlin, Coroutines, and Flow work best")

# Main interface
user_question = st.text_area(
    "Enter your interview question:",
    placeholder="e.g., What is an Activity in Android and what are its lifecycle methods?",
    height=100
)

if st.button("🔍 Get Answer", type="primary"):
    if not user_question.strip():
        st.warning("Please enter a question.")
    else:
        with st.spinner("Searching knowledge base and generating answer..."):
            try:
                # Generate embedding for the question
                query_embedding = st.session_state.embeddings_model.embed_query(user_question)
                
                # Search for similar documents
                retrieved_objects = st.session_state.rag_collection.query.near_vector(
                    near_vector=query_embedding,
                    limit=top_k,
                    return_metadata=wvc.query.MetadataQuery(distance=True)
                )
                
                # Form context from retrieved documents
                retrieved_docs = [obj.properties['content'] for obj in retrieved_objects.objects]
                context = "\n\n---\n\n".join(retrieved_docs)
                
                # Show retrieved sources
                with st.expander("📚 Retrieved Sources", expanded=False):
                    for i, obj in enumerate(retrieved_objects.objects, 1):
                        distance = round(obj.metadata.distance, 4)
                        st.markdown(f"**{i}. {obj.properties['title']}** (distance: {distance})")
                        st.caption(obj.properties['content'][:200] + "...")
                
                # Generate two answers: one without RAG, one with RAG
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("### 💬 Answer WITHOUT RAG:")
                    with st.spinner("Generating answer without context..."):
                        answer_no_rag = st.session_state.answer_chain_no_rag.invoke({
                            "question": user_question
                        })
                    st.markdown(answer_no_rag)
                
                with col2:
                    st.markdown("### 💬 Answer WITH RAG:")
                    with st.spinner("Generating answer with context..."):
                        answer_with_rag = st.session_state.answer_chain_with_rag.invoke({
                            "context": context,
                            "question": user_question
                        })
                    st.markdown(answer_with_rag)
                
                # Show comparison note
                st.info("💡 **Comparison:** The left answer uses only the LLM's general knowledge, while the right answer is based on the retrieved documents from your knowledge base.")
                
            except Exception as e:
                st.error(f"Error: {str(e)}")
                st.exception(e)

# Footer
st.markdown("---")
st.markdown("💡 **Tip:** Make sure the main RAG system notebook (`androd-interview-rag.ipynb`) has been run and Weaviate is running.")

