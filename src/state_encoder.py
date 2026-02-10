"""
State encoder for adaptive RAG policy network.
Encodes the current state of the RAG system into a fixed-size vector.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from sentence_transformers import SentenceTransformer
from abc import ABC, abstractmethod


@dataclass
class StateComponents:
    """Components of the RAG state"""
    question_embedding: np.ndarray
    partial_answer_embedding: np.ndarray
    confidence: float
    retrieval_count: int
    step_count: int
    query_reformulations: int
    retrieved_docs_embedding: np.ndarray
    confidence_history: Optional[np.ndarray] = None
    difficulty_estimate: Optional[float] = None


class BaseStateEncoder(ABC):
    """Abstract base class for state encoders"""
    
    @abstractmethod
    def encode(self, **kwargs) -> np.ndarray:
        """Encode state into vector"""
        pass
    
    @abstractmethod
    def get_state_dim(self) -> int:
        """Get dimension of encoded state vector"""
        pass


class AdaptiveRAGStateEncoder(BaseStateEncoder):
    """Main state encoder for adaptive RAG"""
    
    def __init__(self, config, device="cpu"):
        self.config = config
        self.device = device
        
        # Text embedding model for encoding questions and answers
        self.embedding_model = self._initialize_embedding_model()
        self.embedding_dim = self.embedding_model.get_sentence_embedding_dimension()
        
        # Dimensions for different state components
        self.component_dims = self._calculate_component_dims()
        self.state_dim = sum(self.component_dims.values())
        
        # Normalization statistics (updated during training)
        self.stats = {
            "confidence_mean": 0.5,
            "confidence_std": 0.2,
            "retrieval_mean": 2.0,
            "retrieval_std": 1.5,
            "step_mean": 3.0,
            "step_std": 2.0
        }
        
        # Learnable projection layers
        self.projection_layers = self._initialize_projections()
        
    def _initialize_embedding_model(self):
        """Initialize sentence embedding model"""
        model_name = self.config.model.retriever.embedding_model
        return SentenceTransformer(model_name, device=self.device)
    
    def _calculate_component_dims(self) -> Dict[str, int]:
        """Calculate dimensions for each state component"""
        # Fixed sizes for different components
        return {
            "question": self.embedding_dim,  # Original question
            "partial_answer": self.embedding_dim,  # Current partial answer
            "retrieved_docs": 128,  # Aggregated docs embedding (compressed)
            "scalar_features": 8,  # Confidence, counts, etc.
            "confidence_history": 16,  # Recent confidence trend
        }
    
    def _initialize_projections(self):
        """Initialize learnable projection layers"""
        projections = nn.ModuleDict()
        
        # Projection for retrieved documents (compress multiple docs)
        projections["docs_projection"] = nn.Sequential(
            nn.Linear(self.embedding_dim * 5, 256),  # Assume max 5 docs at a time
            nn.ReLU(),
            nn.Linear(256, self.component_dims["retrieved_docs"])
        )
        
        # Projection for confidence history
        projections["history_projection"] = nn.Sequential(
            nn.Linear(10, 32),  # Last 10 confidence values
            nn.ReLU(),
            nn.Linear(32, self.component_dims["confidence_history"])
        )
        
        return projections.to(self.device)
    
    def encode(self, 
               question: str,
               partial_answer: str,
               confidence: float,
               retrieved_docs: List[str],
               step_count: int,
               query_reformulations: int,
               confidence_history: Optional[List[float]] = None,
               question_difficulty: Optional[float] = None,
               **kwargs) -> np.ndarray:
        """
        Encode the current state of the RAG system
        
        Args:
            question: Original user question
            partial_answer: Current partial answer generated so far
            confidence: Current confidence score (0-1)
            retrieved_docs: List of retrieved document texts
            step_count: Number of steps taken so far
            query_reformulations: Number of query reformulations attempted
            confidence_history: History of confidence values
            question_difficulty: Estimated difficulty of the question
            
        Returns:
            state_vector: Encoded state as numpy array
        """
        # Extract state components
        state_components = self._extract_components(
            question=question,
            partial_answer=partial_answer,
            confidence=confidence,
            retrieved_docs=retrieved_docs,
            step_count=step_count,
            query_reformulations=query_reformulations,
            confidence_history=confidence_history,
            question_difficulty=question_difficulty
        )
        
        # Encode each component
        encoded_components = self._encode_components(state_components)
        
        # Concatenate all components
        state_vector = self._concatenate_components(encoded_components)
        
        return state_vector
    
    def _extract_components(self, **kwargs) -> StateComponents:
        """Extract and compute state components"""
        
        # Encode text components
        question_embedding = self._encode_text(kwargs["question"])
        partial_answer_embedding = self._encode_text(kwargs["partial_answer"])
        
        # Aggregate retrieved documents
        retrieved_docs_embedding = self._encode_retrieved_docs(kwargs["retrieved_docs"])
        
        # Process confidence history
        confidence_history = None
        if kwargs["confidence_history"]:
            confidence_history = np.array(kwargs["confidence_history"][-10:])  # Last 10
            if len(confidence_history) < 10:
                confidence_history = np.pad(confidence_history, 
                                          (0, 10 - len(confidence_history)), 
                                          mode="constant", 
                                          constant_values=0.5)
        
        return StateComponents(
            question_embedding=question_embedding,
            partial_answer_embedding=partial_answer_embedding,
            confidence=kwargs["confidence"],
            retrieval_count=len(kwargs["retrieved_docs"]),
            step_count=kwargs["step_count"],
            query_reformulations=kwargs["query_reformulations"],
            retrieved_docs_embedding=retrieved_docs_embedding,
            confidence_history=confidence_history,
            difficulty_estimate=kwargs.get("question_difficulty")
        )
    
    def _encode_text(self, text: str) -> np.ndarray:
        """Encode text using sentence transformer"""
        if not text:
            # Return zero vector for empty text
            return np.zeros(self.embedding_dim)
        
        with torch.no_grad():
            embedding = self.embedding_model.encode(
                text, 
                convert_to_tensor=True,
                device=self.device
            )
        return embedding.cpu().numpy()
    
    def _encode_retrieved_docs(self, docs: List[str], max_docs: int = 5) -> np.ndarray:
        if not docs:
            # Return zero embedding
            flattened = np.zeros(self.embedding_dim * max_docs)
        else:
            # Encode up to max_docs documents
            doc_embeddings = []
            for i in range(max_docs):
                if i < len(docs):
                    doc_embeddings.append(self._encode_text(docs[i]))
                else:
                    doc_embeddings.append(np.zeros(self.embedding_dim))
            
            flattened = np.concatenate(doc_embeddings)
        
        # Project to lower dimension
        with torch.no_grad():
            tensor = torch.FloatTensor(flattened).unsqueeze(0).to(self.device)
            projected = self.projection_layers["docs_projection"](tensor)
        
        return projected.squeeze().cpu().numpy()
    
    def _encode_components(self, components: StateComponents) -> Dict[str, np.ndarray]:
        """Encode individual components"""
        encoded = {}
        
        # 1. Question embedding
        encoded["question"] = components.question_embedding
        
        # 2. Partial answer embedding
        encoded["partial_answer"] = components.partial_answer_embedding
        
        # 3. Retrieved documents embedding
        encoded["retrieved_docs"] = components.retrieved_docs_embedding
        
        # 4. Scalar features (normalized) - NO difficulty!
        scalar_features = np.array([
            self._normalize(components.confidence, "confidence"),
            self._normalize(components.retrieval_count, "retrieval"),
            self._normalize(components.step_count, "step"),
            components.query_reformulations / 5.0,  # Max 5 reformulations
            len(components.confidence_history) / 10.0 if components.confidence_history is not None else 0.5,
            # Add interaction features
            components.confidence * (components.step_count / 10.0),
            min(1.0, components.retrieval_count / 10.0),
            # Add one more useful feature to maintain array size
            components.confidence * components.retrieval_count / 10.0  # Confidence-retrieval interaction
        ])
        
        # Ensure correct size
        if len(scalar_features) < self.component_dims["scalar_features"]:
            scalar_features = np.pad(scalar_features, 
                                   (0, self.component_dims["scalar_features"] - len(scalar_features)),
                                   mode="constant")
        
        encoded["scalar_features"] = scalar_features[:self.component_dims["scalar_features"]]
        
        # 5. Confidence history
        if components.confidence_history is not None:
            with torch.no_grad():
                tensor = torch.FloatTensor(components.confidence_history).unsqueeze(0).to(self.device)
                projected = self.projection_layers["history_projection"](tensor)
            encoded["confidence_history"] = projected.squeeze().cpu().numpy()
        else:
            encoded["confidence_history"] = np.zeros(self.component_dims["confidence_history"])
        
        return encoded
    
    def _normalize(self, value: float, stat_name: str) -> float:
        """Normalize scalar value using stored statistics"""
        mean = self.stats.get(f"{stat_name}_mean", 0.0)
        std = self.stats.get(f"{stat_name}_std", 1.0)
        
        if std == 0:
            return 0.0
        
        normalized = (value - mean) / std
        # Clip to reasonable range
        return np.clip(normalized, -3.0, 3.0)
    
    def _concatenate_components(self, encoded_components: Dict[str, np.ndarray]) -> np.ndarray:
        """Concatenate all encoded components into single vector"""
        # Concatenate in fixed order
        order = ["question", "partial_answer", "retrieved_docs", 
        "scalar_features", "confidence_history"]
        
        concatenated = []
        for key in order:
            if key in encoded_components:
                concatenated.append(encoded_components[key])
        
        state_vector = np.concatenate(concatenated)
        
        # Ensure correct dimensionality
        if len(state_vector) != self.state_dim:
            # Pad or truncate if needed
            if len(state_vector) < self.state_dim:
                state_vector = np.pad(state_vector, 
                                    (0, self.state_dim - len(state_vector)),
                                    mode="constant")
            else:
                state_vector = state_vector[:self.state_dim]
        
        return state_vector
    
    def get_state_dim(self) -> int:
        """Get the dimension of the encoded state vector"""
        return self.state_dim
    
    def update_statistics(self, 
                         confidences: List[float],
                         retrieval_counts: List[int],
                         step_counts: List[int]):
        """Update normalization statistics from training data"""
        if confidences:
            self.stats["confidence_mean"] = np.mean(confidences)
            self.stats["confidence_std"] = np.std(confidences) or 0.2
        
        if retrieval_counts:
            self.stats["retrieval_mean"] = np.mean(retrieval_counts)
            self.stats["retrieval_std"] = np.std(retrieval_counts) or 1.5
        
        if step_counts:
            self.stats["step_mean"] = np.mean(step_counts)
            self.stats["step_std"] = np.std(step_counts) or 2.0
    
    def save(self, path: str):
        """Save encoder state"""
        state = {
            "config": self.config,
            "stats": self.stats,
            "projection_state_dict": self.projection_layers.state_dict(),
            "component_dims": self.component_dims
        }
        torch.save(state, path)
    
    def load(self, path: str):
        """Load encoder state"""
        state = torch.load(path, map_location=self.device)
        self.stats = state["stats"]
        self.component_dims = state["component_dims"]
        self.projection_layers.load_state_dict(state["projection_state_dict"])

class LightweightStateEncoder(BaseStateEncoder):
    """Lightweight state encoder for faster inference"""
    
    def __init__(self, config, device='cpu'):
        self.config = config
        self.device = device
        self.state_dim = 64  # Fixed small dimension
        
        # Simple embedding projection
        self.embedding_dim = 384  # Smaller fixed embedding
        self.projection = nn.Sequential(
            nn.Linear(self.embedding_dim * 2 + 6, 128),
            nn.ReLU(),
            nn.Linear(128, self.state_dim)
        ).to(device)
    
    def encode(self, **kwargs) -> np.ndarray:
        """Simple encoding for lightweight deployment"""
        # Extract key features
        features = []
        
        # Text length features (proxy for complexity)
        features.append(min(1.0, len(kwargs.get("question", "")) / 100.0))
        features.append(min(1.0, len(kwargs.get("partial_answer", "")) / 500.0))
        
        # Count features
        features.append(min(1.0, kwargs.get("step_count", 0) / 10.0))
        features.append(min(1.0, kwargs.get("query_reformulations", 0) / 5.0))
        features.append(min(1.0, len(kwargs.get("retrieved_docs", [])) / 10.0))
        
        # Confidence
        features.append(kwargs.get("confidence", 0.5))
        
        # Simple text embeddings
        question_sim = hash(kwargs.get("question", "")) % 1000 / 1000.0
        answer_sim = hash(kwargs.get("partial_answer", "")) % 1000 / 1000.0
        
        features.append(question_sim)
        features.append(answer_sim)
        
        # Convert to tensor and project
        with torch.no_grad():
            tensor = torch.FloatTensor(features).unsqueeze(0).to(self.device)
            # Pad to expected input size
            if tensor.size(1) < self.embedding_dim * 2 + 6:
                padding = torch.zeros(1, self.embedding_dim * 2 + 6 - tensor.size(1)).to(self.device)
                tensor = torch.cat([tensor, padding], dim=1)
            
            encoded = self.projection(tensor)
        
        return encoded.squeeze().cpu().numpy()
    
    def get_state_dim(self) -> int:
        return self.state_dim