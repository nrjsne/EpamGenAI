import os
from langchain_core.messages import AIMessage
from langchain_core.runnables import Runnable, RunnableConfig
from sentence_transformers import SentenceTransformer
from transformers import pipeline
import torch


class LocalHuggingFaceEmbeddings:
    """Wrapper for local embedding model."""
    
    def __init__(self, model_name):
        print(f"📥 Loading local embedding model: {model_name}...")
        try:
            hf_token = os.environ.get("HUGGINGFACE_API_TOKEN", None)
            if hf_token:
                self.model = SentenceTransformer(model_name, token=hf_token)
            else:
                self.model = SentenceTransformer(model_name)
            print("✅ Local embedding model loaded successfully.")
        except Exception as e:
            print(f"❌ Error loading {model_name}. Falling back to 'all-MiniLM-L6-v2'.")
            print(f"Error details: {e}")
            self.model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    def embed_documents(self, texts):
        embeddings = self.model.encode(texts, convert_to_numpy=True)
        return embeddings.tolist()

    def embed_query(self, text):
        embedding = self.model.encode(text, convert_to_numpy=True)
        return embedding.tolist()


class LocalHuggingFaceChatModel(Runnable):
    """Wrapper for local LLM."""
    
    def __init__(self, model_name):
        print(f"📥 Loading local LLM: {model_name}...")
        
        token = os.environ.get("HUGGINGFACE_API_TOKEN", None)
        try:
            pipeline_kwargs = {
                "task": "text-generation",
                "model": model_name,
                "device": -1,  # CPU
                "torch_dtype": torch.float32
            }
            
            if token:
                pipeline_kwargs["token"] = token
            
            self.pipe = pipeline(**pipeline_kwargs)
            print("✅ Local LLM loaded successfully.")
        except Exception as e:
            print(f"❌ Error loading {model_name}: {e}")
            print("💡 Tip: Make sure you're logged in via 'huggingface-cli login' if the model requires authentication.")
            raise

    def invoke(self, input_data, config: RunnableConfig = None, **kwargs):
        messages = []

        if hasattr(input_data, 'to_messages'):
            lc_messages = input_data.to_messages()
            for msg in lc_messages:
                role = "user"
                if msg.type == "system": 
                    role = "system"
                elif msg.type == "ai": 
                    role = "assistant"
                messages.append({"role": role, "content": [{"type": "text", "text": msg.content}]})
        elif isinstance(input_data, str):
            messages = [{"role": "user", "content": [{"type": "text", "text": input_data}]}]

        # Use deterministic generation for reproducibility
        # Set random seed for reproducibility (if torch is available)
        try:
            torch.manual_seed(42)
        except:
            pass  # Ignore if torch seed setting fails
        
        # Use do_sample=False for greedy/deterministic decoding
        # According to error message: "If you're looking for greedy decoding strategies, set do_sample=False"
        generation_kwargs = {
            "max_new_tokens": 512,
            "do_sample": False  # Greedy decoding for deterministic output
        }
        
        # Try with do_sample=False, fallback if not supported
        try:
            outputs = self.pipe(messages, **generation_kwargs)
        except (ValueError, TypeError) as e:
            error_str = str(e).lower()
            # If do_sample is not supported, try with just max_new_tokens
            if "model_kwargs" in error_str or "not used" in error_str or "not supported" in error_str or "do_sample" in error_str:
                generation_kwargs = {"max_new_tokens": 512}
                try:
                    outputs = self.pipe(messages, **generation_kwargs)
                except:
                    # Last resort: minimal kwargs
                    outputs = self.pipe(messages)
            else:
                raise
        generated_text = outputs[0]['generated_text'][-1]['content']
        return AIMessage(content=generated_text)

