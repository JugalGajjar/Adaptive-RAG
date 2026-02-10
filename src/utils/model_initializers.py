"""
Utility functions for initializing generator and retriever models.
"""

import torch
from typing import Optional, Dict, Any, Union
import openai
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModel
from langchain_openai import OpenAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS, Chroma
from langchain_community.retrievers import BM25Retriever
import faiss
import numpy as np
from pathlib import Path
import logging
import time
from tqdm import tqdm

logger = logging.getLogger(__name__)


def get_best_device():
    """
    Get the best available device (CUDA > MPS > CPU)
    
    Returns:
        str: Device string ('cuda', 'mps', or 'cpu')
    """
    if torch.cuda.is_available():
        device = "cuda"
        device_name = torch.cuda.get_device_name(0)
        print(f"Using CUDA: {device_name}")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
        print(f"Using Apple Silicon MPS")
    else:
        device = "cpu"
        print(f"Using CPU")
    
    return device

def initialize_generator(config: Dict[str, Any]) -> Any:
    """
    Initialize the generator model based on configuration.
    
    Supported types:
    - openai: OpenAI API (GPT-3.5, GPT-4)
    - groq: Groq API (Llama-70B, Qwen-32B, GPT-OSS-120B) - FREE TIER
    - huggingface: HuggingFace models (Llama, Mistral, Qwen) - LOCAL
    - local: Alias for huggingface
    
    Args:
        config: Generator configuration dictionary
        
    Returns:
        Initialized generator object
    """
    generator_type = config.get('type', 'openai').lower()
    
    if generator_type == "openai":
        return _initialize_openai_generator(config)
    elif generator_type == "groq":
        return _initialize_groq_generator(config)
    elif generator_type in ["huggingface", "local"]:
        return _initialize_hf_generator(config)
    else:
        raise ValueError(f"Unknown generator type: {generator_type}")


def _initialize_openai_generator(config: Dict[str, Any]) -> "OpenAIGenerator":
    """Initialize OpenAI API generator"""
    
    class OpenAIGenerator:
        def __init__(self, config):
            self.config = config
            self.model_name = config.get("name", "gpt-3.5-turbo")
            self.max_tokens = config.get("max_tokens", 512)
            self.temperature = config.get("temperature", 0.1)
            
            # Initialize OpenAI client
            api_key = config.get("api_key")
            if not api_key:
                raise ValueError("OpenAI API key not provided")
            
            self.client = openai.OpenAI(api_key=api_key)
        
        def generate(self, prompt: str, **kwargs) -> str:
            """Generate text from prompt"""
            max_tokens = kwargs.get("max_tokens", self.max_tokens)
            temperature = kwargs.get("temperature", self.temperature)
            
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=max_tokens,
                    temperature=temperature,
                    **{k: v for k, v in kwargs.items() 
                       if k not in ["max_tokens", "temperature"]}
                )
                return response.choices[0].message.content
            except Exception as e:
                logger.error(f"OpenAI generation error: {e}")
                return f"Error generating response: {str(e)}"
        
        def generate_with_confidence(self, prompt: str, **kwargs) -> tuple:
            """Generate text with confidence score"""
            # For OpenAI models, we can't get proper confidence scores
            # So we use a heuristic based on response characteristics
            response = self.generate(prompt, **kwargs)
            
            # Simple confidence heuristic
            confidence = self._estimate_confidence(response)
            return response, confidence
        
        def _estimate_confidence(self, response: str) -> float:
            """Estimate confidence based on response characteristics"""
            # Heuristics for confidence estimation
            indicators = {
                "i don\'t know": -0.5,
                "i\'m not sure": -0.5,
                "i cannot": -0.3,
                "unable to": -0.3,
                "based on": 0.2,
                "according to": 0.2,
                "research shows": 0.3,
                "studies indicate": 0.3,
                "definitely": 0.4,
                "certainly": 0.4,
                "clearly": 0.3
            }
            
            response_lower = response.lower()
            confidence = 0.7  # Base confidence
            
            # Adjust based on indicators
            for phrase, adjustment in indicators.items():
                if phrase in response_lower:
                    confidence += adjustment
            
            # Adjust based on response length
            if len(response.split()) < 5:
                confidence -= 0.2
            
            # Adjust based on hedging language
            hedging_words = ["might", "could", "perhaps", "maybe", "possibly"]
            for word in hedging_words:
                if word in response_lower:
                    confidence -= 0.1
            
            return max(0.1, min(1.0, confidence))  # Clip to [0.1, 1.0]
    
    return OpenAIGenerator(config)

def _initialize_groq_generator(config: Dict[str, Any]) -> "GroqGenerator":
    """
    Initialize Groq API generator (FREE TIER AVAILABLE)
    
    Supported models:
    - meta-llama/llama-3.3-70b-versatile (70B)
    - qwen/qwen-2.5-32b-instruct (32B)
    - openai/gpt-oss-120b (120B)
    
    Free tier: 30 requests/minute
    """
    
    class GroqGenerator:
        def __init__(self, config):
            try:
                from groq import Groq
            except ImportError:
                raise ImportError(
                    "Groq library not installed. Install with: pip install groq"
                )
            
            self.config = config
            self.model_name = config.get("name", "meta-llama/llama-3.3-70b-versatile")
            self.max_tokens = config.get("max_tokens", 512)
            self.temperature = config.get("temperature", 0.1)
            
            # Initialize Groq client
            api_key = config.get("api_key")
            if not api_key:
                raise ValueError("Groq API key not provided. Set GROQ_API_KEY environment variable.")
            
            self.client = Groq(api_key=api_key)
            
            # Rate limiting (free tier: 30 req/min)
            self.requests_per_minute = config.get("requests_per_minute", 25)  # Stay under limit
            self.request_interval = 60.0 / self.requests_per_minute
            self.last_request_time = 0
            
            # Retry settings
            self.max_retries = config.get("max_retries", 3)
            self.retry_on_rate_limit = config.get("retry_on_rate_limit", True)
            
            logger.info(f"Initialized Groq generator: {self.model_name}")
            logger.info(f"Rate limit: {self.requests_per_minute} req/min")
        
        def _wait_for_rate_limit(self):
            """Enforce rate limiting"""
            current_time = time.time()
            time_since_last = current_time - self.last_request_time
            
            if time_since_last < self.request_interval:
                sleep_time = self.request_interval - time_since_last
                logger.debug(f"Rate limiting: sleeping {sleep_time:.2f}s")
                time.sleep(sleep_time)
        
        def generate(self, prompt: str, **kwargs) -> str:
            """Generate text from prompt with rate limiting"""
            max_tokens = kwargs.get("max_tokens", self.max_tokens)
            temperature = kwargs.get("temperature", self.temperature)
            
            # Apply rate limiting
            self._wait_for_rate_limit()
            
            # Retry logic
            for attempt in range(self.max_retries):
                try:
                    response = self.client.chat.completions.create(
                        model=self.model_name,
                        messages=[{"role": "user", "content": prompt}],
                        max_tokens=max_tokens,
                        temperature=temperature,
                        **{k: v for k, v in kwargs.items() 
                           if k not in ["max_tokens", "temperature"]}
                    )
                    
                    self.last_request_time = time.time()
                    return response.choices[0].message.content.strip()
                    
                except Exception as e:
                    error_str = str(e).lower()
                    
                    # Handle rate limit errors
                    if "rate_limit" in error_str or "429" in error_str:
                        if self.retry_on_rate_limit and attempt < self.max_retries - 1:
                            wait_time = 5 * (attempt + 1)  # Exponential backoff
                            logger.warning(f"Rate limit hit, waiting {wait_time}s (attempt {attempt + 1}/{self.max_retries})")
                            time.sleep(wait_time)
                            continue
                        else:
                            logger.error("Rate limit exceeded and max retries reached")
                            raise
                    
                    # Handle other errors
                    logger.error(f"Groq generation error (attempt {attempt + 1}/{self.max_retries}): {e}")
                    
                    if attempt < self.max_retries - 1:
                        time.sleep(2 * (attempt + 1))
                        continue
                    else:
                        return f"Error generating response: {str(e)}"
            
            return "Error: Max retries exceeded"
        
        def generate_with_confidence(self, prompt: str, **kwargs) -> tuple:
            """Generate text with confidence score"""
            response = self.generate(prompt, **kwargs)
            confidence = self._estimate_confidence(response)
            return response, confidence
        
        def _estimate_confidence(self, response: str) -> float:
            """Estimate confidence based on response characteristics"""
            indicators = {
                "i don\'t know": -0.5,
                "i\'m not sure": -0.5,
                "i cannot": -0.3,
                "unable to": -0.3,
                "based on": 0.2,
                "according to": 0.2,
                "research shows": 0.3,
                "definitely": 0.4,
                "certainly": 0.4,
                "clearly": 0.3
            }
            
            response_lower = response.lower()
            confidence = 0.7
            
            for phrase, adjustment in indicators.items():
                if phrase in response_lower:
                    confidence += adjustment
            
            if len(response.split()) < 5:
                confidence -= 0.2
            
            hedging_words = ["might", "could", "perhaps", "maybe", "possibly"]
            for word in hedging_words:
                if word in response_lower:
                    confidence -= 0.1
            
            return max(0.1, min(1.0, confidence))
    
    return GroqGenerator(config)

def _initialize_hf_generator(config: Dict[str, Any]) -> "HFGenerator":
    """
    Initialize HuggingFace generator for local models
    
    Supported models:
    - meta-llama/Llama-3.1-8B-Instruct (8B)
    - Qwen/Qwen3-4B-Instruct-v0.3 (4B)
    - Qwen/Qwen2.5-7B-Instruct (7B)
    - microsoft/Phi-3.5-mini-instruct (3.8B)
    """
    
    class HFGenerator:
        def __init__(self, config):
            self.config = config
            self.model_name = config.get("name", "microsoft/Phi-3.5-mini-instruct")
            self.device = config.get("device", get_best_device())
            
            logger.info(f"Loading HuggingFace model: {self.model_name}")
            logger.info(f"Device: {self.device}")
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=True  # For models like Qwen
            )
            
            # Model loading configuration
            torch_dtype = config.get("torch_dtype", "float16")
            load_in_8bit = config.get("load_in_8bit", False)
            load_in_4bit = config.get("load_in_4bit", False)
            
            # Determine dtype
            if torch_dtype == "float16":
                dtype = torch.float16
            elif torch_dtype == "bfloat16":
                dtype = torch.bfloat16
            else:
                dtype = torch.float32
            
            # Load model with appropriate configuration
            model_kwargs = {"trust_remote_code": True}
            
            if self.device == "cuda":
                model_kwargs["device_map"] = "auto"
                model_kwargs["torch_dtype"] = dtype
                
                if load_in_8bit:
                    logger.info("Loading in 8-bit mode")
                    model_kwargs["load_in_8bit"] = True
                elif load_in_4bit:
                    logger.info("Loading in 4-bit mode")
                    model_kwargs["load_in_4bit"] = True
            
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                **model_kwargs
            )
            
            # Move to device if not using device_map
            if self.device != "cuda" or "device_map" not in model_kwargs:
                self.model = self.model.to(self.device)
            
            # Set padding token if not set
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
            
            self.max_tokens = config.get("max_tokens", 512)
            self.temperature = config.get("temperature", 0.1)
            
            logger.info(f"Model loaded successfully on {self.device}")
        
        def generate(self, prompt: str, **kwargs) -> str:
            """Generate text from prompt"""
            max_tokens = kwargs.get("max_tokens", self.max_tokens)
            temperature = kwargs.get("temperature", self.temperature)
            
            # Tokenize input
            inputs = self.tokenizer(
                prompt, 
                return_tensors="pt", 
                truncation=True, 
                max_length=2048
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Generation parameters
            gen_kwargs = {
                "max_new_tokens": max_tokens,
                "temperature": max(temperature, 0.01),
                "do_sample": temperature > 0.01,
                "pad_token_id": self.tokenizer.pad_token_id,
                "eos_token_id": self.tokenizer.eos_token_id,
            }
            
            # Add any additional kwargs
            for k, v in kwargs.items():
                if k not in ["max_tokens", "temperature"]:
                    gen_kwargs[k] = v
            
            # Generate
            with torch.no_grad():
                outputs = self.model.generate(**inputs, **gen_kwargs)
            
            # Decode
            full_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Remove the prompt from the response
            if full_response.startswith(prompt):
                response = full_response[len(prompt):].strip()
            else:
                response = full_response.strip()
            
            return response
        
        def generate_with_confidence(self, prompt: str, **kwargs) -> tuple:
            """Generate text with confidence score using token probabilities"""
            max_tokens = kwargs.get("max_tokens", self.max_tokens)
            temperature = kwargs.get("temperature", self.temperature)
            
            # Tokenize input
            inputs = self.tokenizer(
                prompt, 
                return_tensors="pt", 
                truncation=True, 
                max_length=2048
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Generate with scores
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    temperature=max(temperature, 0.01),
                    do_sample=temperature > 0.01,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    output_scores=True,
                    return_dict_in_generate=True
                )
            
            # Decode response
            full_response = self.tokenizer.decode(outputs.sequences[0], skip_special_tokens=True)
            if full_response.startswith(prompt):
                response = full_response[len(prompt):].strip()
            else:
                response = full_response.strip()
            
            # Calculate confidence from token probabilities
            if hasattr(outputs, "scores") and outputs.scores:
                token_probs = []
                for score in outputs.scores:
                    probs = torch.softmax(score[0], dim=-1)
                    max_prob = probs.max().item()
                    token_probs.append(max_prob)
                
                if token_probs:
                    confidence = np.mean(token_probs)
                else:
                    confidence = 0.5
            else:
                confidence = self._estimate_confidence_heuristic(response)
            
            return response, confidence
        
        def _estimate_confidence_heuristic(self, response: str) -> float:
            """Fallback heuristic confidence estimation"""
            indicators = {
                "i don\'t know": -0.5,
                "i\'m not sure": -0.5,
                "i cannot": -0.3,
                "unable to": -0.3,
                "based on": 0.2,
                "according to": 0.2,
                "clearly": 0.3,
                "definitely": 0.4
            }
            
            response_lower = response.lower()
            confidence = 0.7
            
            for phrase, adjustment in indicators.items():
                if phrase in response_lower:
                    confidence += adjustment
            
            if len(response.split()) < 5:
                confidence -= 0.2
            
            return max(0.1, min(1.0, confidence))
    
    return HFGenerator(config)


def _initialize_local_generator(config: Dict[str, Any]) -> "LocalGenerator":
    """Initialize a simple local generator (for testing)"""
    
    class LocalGenerator:
        def __init__(self, config):
            self.config = config
            self.responses = config.get("responses", {})
            
        def generate(self, prompt: str, **kwargs) -> str:
            """Generate a simple response based on prompt keywords"""
            prompt_lower = prompt.lower()
            
            # Check for keywords in prompt
            for keyword, response in self.responses.items():
                if keyword in prompt_lower:
                    return response
            
            # Default response
            return f"Based on the information provided: {prompt[:50]}..."
        
        def generate_with_confidence(self, prompt: str, **kwargs) -> tuple:
            response = self.generate(prompt, **kwargs)
            
            # Simple confidence based on response length
            confidence = min(1.0, len(response) / 100.0)
            return response, confidence
    
    return LocalGenerator(config)


def initialize_retriever(config: Dict[str, Any], corpus: list = None) -> Any:
    """
    Initialize the retriever based on configuration.
    
    Args:
        config: Retriever configuration dictionary
        corpus: List of documents for the retriever
        
    Returns:
        Initialized retriever object
    """
    retriever_type = config.get("type", "dense").lower()
    
    if retriever_type == "dense":
        return _initialize_dense_retriever(config, corpus)
    elif retriever_type == "sparse":
        return _initialize_sparse_retriever(config, corpus)
    elif retriever_type == "hybrid":
        return _initialize_hybrid_retriever(config, corpus)
    elif retriever_type == "mock":
        return _initialize_mock_retriever(config, corpus)
    else:
        raise ValueError(f"Unknown retriever type: {retriever_type}")


def _initialize_dense_retriever(config: Dict[str, Any], corpus: list) -> "DenseRetriever":
    """Initialize dense retriever (vector similarity)"""
    
    class DenseRetriever:
        def __init__(self, config, corpus):
            self.config = config
            self.corpus = corpus or []
            self.top_k = config.get("top_k_initial", 3)
            
            # Initialize embeddings
            embedding_model_name = config.get("embedding_model", "all-MiniLM-L6-v2")
            
            if config.get("use_openai_embeddings", False):
                api_key = config.get("openai_api_key")
                if not api_key:
                    raise ValueError("OpenAI API key required for OpenAI embeddings")
                self.embeddings = OpenAIEmbeddings(
                    openai_api_key=api_key,
                    model="text-embedding-ada-002"
                )
            else:
                self.embeddings = HuggingFaceEmbeddings(
                    model_name=embedding_model_name,
                    model_kwargs={"device": "cpu"},
                    encode_kwargs={"normalize_embeddings": True}
                )
            
            # Create vector store
            self.vector_store = self._create_vector_store()
        
        def _create_vector_store(self):
            """Create vector store from corpus"""
            if not self.corpus:
                logger.warning("No corpus provided for retriever")
                return None
            
            # Extract texts from corpus
            texts = []
            metadatas = []
            
            for doc in self.corpus:
                if hasattr(doc, 'text'):
                    texts.append(doc.text)
                    metadata = {
                        "id": getattr(doc, "id", "unknown"),
                        "title": getattr(doc, "title", "")
                    }
                    if hasattr(doc, "metadata"):
                        metadata.update(doc.metadata)
                    metadatas.append(metadata)
                elif isinstance(doc, dict):
                    texts.append(doc.get("text", ""))
                    metadatas.append(doc.get("metadata", {}))
                else:
                    texts.append(str(doc))
                    metadatas.append({})
            
            # Create FAISS vector store
            vector_store = FAISS.from_texts(
                texts=texts,
                embedding=self.embeddings,
                metadatas=metadatas
            )
            
            return vector_store
        
        def retrieve(self, query: str, top_k: int = None, **kwargs) -> list:
            """Retrieve documents for query"""
            if not self.vector_store:
                return []
            
            k = top_k or self.top_k
            offset = kwargs.get("offset", 0)
            
            # Get documents with similarity scores
            docs_with_scores = self.vector_store.similarity_search_with_score(
                query, 
                k=k + offset
            )
            
            # Apply offset if specified
            if offset > 0:
                docs_with_scores = docs_with_scores[offset:]
            
            # Format results
            results = []
            for doc, score in docs_with_scores:
                results.append({
                    "text": doc.page_content,
                    "score": float(score),
                    "metadata": doc.metadata
                })
            
            return results
        
        def retrieve_with_embedding(self, query_embedding: np.ndarray, top_k: int = None) -> list:
            """Retrieve using pre-computed query embedding"""
            if not self.vector_store:
                return []
            
            k = top_k or self.top_k
            
            # Convert to float32 for FAISS
            query_embedding = query_embedding.astype(np.float32).reshape(1, -1)
            
            # Get index from vector store
            index = self.vector_store.index
            vector_dim = index.d
            
            # Ensure embedding dimension matches
            if query_embedding.shape[1] != vector_dim:
                logger.warning(f"Embedding dimension mismatch: {query_embedding.shape[1]} != {vector_dim}")
                return []
            
            # Search
            distances, indices = index.search(query_embedding, k)
            
            # Get documents
            results = []
            for i, (dist, idx) in enumerate(zip(distances[0], indices[0])):
                if idx < len(self.vector_store.docstore._dict):
                    doc = self.vector_store.docstore._dict[idx]
                    results.append({
                        "text": doc.page_content,
                        "score": float(1 / (1 + dist)),  # Convert distance to similarity
                        "metadata": doc.metadata
                    })
            
            return results
    
    return DenseRetriever(config, corpus)


def _initialize_sparse_retriever(config: Dict[str, Any], corpus: list) -> "SparseRetriever":
    """Initialize sparse retriever (BM25)"""
    
    class SparseRetriever:
        def __init__(self, config, corpus):
            self.config = config
            self.corpus = corpus or []
            self.top_k = config.get("top_k_initial", 3)
            
            # Create BM25 retriever
            self.retriever = self._create_bm25_retriever()
        
        def _create_bm25_retriever(self):
            """Create BM25 retriever from corpus"""
            if not self.corpus:
                return None
            
            # Extract texts
            texts = []
            for doc in self.corpus:
                if hasattr(doc, "text"):
                    texts.append(doc.text)
                elif isinstance(doc, dict):
                    texts.append(doc.get("text", ""))
                else:
                    texts.append(str(doc))
            
            # Create BM25 retriever
            from rank_bm25 import BM25Okapi
            import nltk
            
            # Tokenize texts
            try:
                nltk.data.find("tokenizers/punkt")
            except LookupError:
                nltk.download("punkt")
            
            tokenized_texts = [nltk.word_tokenize(text.lower()) for text in texts]
            bm25 = BM25Okapi(tokenized_texts)
            
            return bm25
        
        def retrieve(self, query: str, top_k: int = None, **kwargs) -> list:
            """Retrieve documents using BM25"""
            if not self.retriever:
                return []
            
            k = top_k or self.top_k
            offset = kwargs.get("offset", 0)
            
            # Tokenize query
            import nltk
            tokenized_query = nltk.word_tokenize(query.lower())
            
            # Get scores
            scores = self.retriever.get_scores(tokenized_query)
            
            # Get top-k indices
            total_docs = len(self.corpus)
            k_total = min(k + offset, total_docs)
            
            if k_total <= 0:
                return []
            
            # Get indices sorted by score
            indices = np.argsort(scores)[::-1][:k_total]
            
            # Apply offset
            if offset > 0:
                indices = indices[offset:]
            
            # Format results
            results = []
            for idx in indices:
                doc = self.corpus[idx]
                score = scores[idx]
                
                if hasattr(doc, "text"):
                    text = doc.text
                    metadata = {"id": getattr(doc, "id", str(idx))}
                elif isinstance(doc, dict):
                    text = doc.get("text", "")
                    metadata = doc.get("metadata", {})
                else:
                    text = str(doc)
                    metadata = {}
                
                results.append({
                    "text": text,
                    "score": float(score),
                    "metadata": metadata
                })
            
            return results
    
    return SparseRetriever(config, corpus)


def _initialize_hybrid_retriever(config: Dict[str, Any], corpus: list) -> "HybridRetriever":
    """Initialize hybrid retriever (dense + sparse)"""
    
    class HybridRetriever:
        def __init__(self, config, corpus):
            self.config = config
            self.corpus = corpus or []
            
            # Initialize both retrievers
            dense_config = {**config, "type": "dense"}
            sparse_config = {**config, "type": "sparse"}
            
            self.dense_retriever = _initialize_dense_retriever(dense_config, corpus)
            self.sparse_retriever = _initialize_sparse_retriever(sparse_config, corpus)
            
            # Hybrid weights
            self.dense_weight = config.get("dense_weight", 0.7)
            self.sparse_weight = config.get("sparse_weight", 0.3)
            self.top_k = config.get("top_k_initial", 3)
        
        def retrieve(self, query: str, top_k: int = None, **kwargs) -> list:
            """Retrieve documents using hybrid approach"""
            k = top_k or self.top_k
            
            # Get results from both retrievers
            dense_results = self.dense_retriever.retrieve(query, k * 2, **kwargs)
            sparse_results = self.sparse_retriever.retrieve(query, k * 2, **kwargs)
            
            # Create document to score mapping
            doc_scores = {}
            
            # Add dense results
            for result in dense_results:
                doc_text = result["text"]
                if doc_text not in doc_scores:
                    doc_scores[doc_text] = {
                        "text": doc_text,
                        "metadata": result["metadata"],
                        "dense_score": result["score"],
                        "sparse_score": 0.0
                    }
                else:
                    doc_scores[doc_text]["dense_score"] = result["score"]
            
            # Add sparse results
            for result in sparse_results:
                doc_text = result["text"]
                if doc_text not in doc_scores:
                    doc_scores[doc_text] = {
                        "text": doc_text,
                        "metadata": result["metadata"],
                        "dense_score": 0.0,
                        "sparse_score": result["score"]
                    }
                else:
                    doc_scores[doc_text]["sparse_score"] = result["score"]
            
            # Calculate hybrid scores
            scored_docs = []
            for doc_info in doc_scores.values():
                # Normalize scores if needed
                dense_score = doc_info["dense_score"]
                sparse_score = doc_info["sparse_score"]
                
                # Simple weighted combination
                hybrid_score = (self.dense_weight * dense_score + 
                              self.sparse_weight * sparse_score)
                
                scored_docs.append({
                    "text": doc_info["text"],
                    "score": hybrid_score,
                    "metadata": doc_info["metadata"],
                    "dense_score": dense_score,
                    "sparse_score": sparse_score
                })
            
            # Sort by hybrid score
            scored_docs.sort(key=lambda x: x["score"], reverse=True)
            
            # Return top-k
            return scored_docs[:k]
    
    return HybridRetriever(config, corpus)


def _initialize_mock_retriever(config: Dict[str, Any], corpus: list) -> "MockRetriever":
    """Initialize mock retriever for testing"""
    
    class MockRetriever:
        def __init__(self, config, corpus):
            self.config = config
            self.corpus = corpus or []
            self.top_k = config.get("top_k_initial", 3)
            
            # Create mock responses based on query keywords
            self.mock_responses = {
                "capital": ["Paris is the capital of France.", "Tokyo is the capital of Japan."],
                "science": ["Einstein developed the theory of relativity.", "Newton discovered gravity."],
                "history": ["World War II ended in 1945.", "The Renaissance began in Italy."],
                "default": ["This is a relevant document.", "Here is some information."]
            }
        
        def retrieve(self, query: str, top_k: int = None, **kwargs) -> list:
            """Return mock documents based on query"""
            k = top_k or self.top_k
            
            # Find matching keyword
            query_lower = query.lower()
            matching_keyword = "default"
            
            for keyword in self.mock_responses:
                if keyword in query_lower:
                    matching_keyword = keyword
                    break
            
            # Get mock documents
            mock_docs = self.mock_responses.get(matching_keyword, self.mock_responses["default"])
            
            # Format as retrieval results
            results = []
            for i, doc in enumerate(mock_docs[:k]):
                results.append({
                    "text": doc,
                    "score": 0.9 - (i * 0.1),  # Decreasing scores
                    "metadata": {"mock": True, "keyword": matching_keyword}
                })
            
            return results
    
    return MockRetriever(config, corpus)