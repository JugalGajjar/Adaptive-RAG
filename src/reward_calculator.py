import numpy as np
from typing import List, Dict, Any
from dataclasses import dataclass
from abc import ABC, abstractmethod
import torch
import logging
logger = logging.getLogger(__name__)

from sentence_transformers import SentenceTransformer, util
import torch

@dataclass
class RewardComponents:
    correctness: float
    retrieval_penalty: float
    latency_penalty: float
    step_penalty: float
    retrieval_quality: float = 0.5
    confidence_gain: float = 0.0
    document_novelty: float = 0.0

class RewardCalculator(ABC):
    """Base class for reward calculation"""
    
    @abstractmethod
    def calculate(self, question: str, answer: str, **kwargs) -> float:
        pass

class AdaptiveRAGRewardCalculator(RewardCalculator):
    def __init__(self, config, answer_evaluator=None):
        self.config = config
        self.weights = config.training.reward_weights
        self.answer_evaluator = answer_evaluator or ExactMatchEvaluator()

        # Initialize embedding model
        try:
            from sentence_transformers import SentenceTransformer
            self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
        except Exception as e:
            logger.warning(f"Could not load embedding model: {e}")
            self.embedding_model = None
        
        # New weight for retrieval quality
        self.weights.retrieval_quality = config.training.reward_weights.get("retrieval_quality", 2.0)
    
    def calculate(self, question: str, answer: str, **kwargs) -> float:
        """
        Calculate reward based on OUTCOME and EFFICIENCY (not difficulty rules).
        Rewards achieving high confidence efficiently, regardless of question difficulty.
        """
        # 1. Correctness (primary goal)
        reference = kwargs.get("reference_answer")
        correct = self.answer_evaluator.evaluate(question, answer, reference)
        correctness_reward = self.weights.correctness if correct else 0.0
        
        # 2. Efficiency: confidence gain per retrieval
        retrieval_count = kwargs.get("retrieval_count", 0)
        confidence_gain = kwargs.get("confidence_gain", 0)
        
        if retrieval_count > 0:
            efficiency = confidence_gain / retrieval_count
            efficiency_reward = efficiency * 2.0  # Reward efficient learning
        else:
            efficiency_reward = 0
        
        # 3. Over-retrieval penalty
        # If started with high confidence but still retrieved a lot = wasteful
        initial_confidence = kwargs.get("initial_confidence", 0.5)
        final_confidence = kwargs.get("final_confidence", 0.5)
        
        if initial_confidence > 0.75 and retrieval_count > 2:
            waste_penalty = -0.4 * (retrieval_count - 2)
        else:
            waste_penalty = 0
        
        # 4. Under-retrieval penalty
        # If ended with low confidence and didn't retrieve much = lazy
        if final_confidence < 0.6 and retrieval_count < 2:
            underretrieval_penalty = -0.5
        else:
            underretrieval_penalty = 0
        
        # 5. Confidence achievement bonus
        if final_confidence > 0.85:
            confidence_bonus = 1.0
        elif final_confidence > 0.75:
            confidence_bonus = 0.5
        else:
            confidence_bonus = 0
        
        # 6. Basic costs
        retrieval_penalty = self.weights.retrieval_penalty * retrieval_count
        latency_penalty = -min(kwargs.get("latency", 0) / 10.0, 1.0)
        step_penalty = self.weights.step_penalty * kwargs.get("step_count", 0)
        
        # 7. Retrieval quality bonus
        retrieval_quality = kwargs.get("retrieval_quality", 0.5)
        quality_bonus = (retrieval_quality - 0.5) * 1.0  # -0.5 to +0.5
        
        total_reward = (
            correctness_reward +        # 5.0 if correct, 0 if wrong
            efficiency_reward +         # 0-2.0 based on conf gain per retrieval
            confidence_bonus +          # 0-1.0 based on final confidence
            waste_penalty +             # 0 to -1.2 if over-retrieved
            underretrieval_penalty +    # -0.5 if under-retrieved
            retrieval_penalty +         # -0.1 per retrieval
            latency_penalty +           # -0.0 to -1.0 based on time
            step_penalty +              # -0.05 per step
            quality_bonus               # -0.5 to +0.5 based on doc relevance
        )
        
        return total_reward
    
    def calculate_intermediate(self, confidence_gain: float, **kwargs) -> float:
        """Calculate intermediate reward with retrieval quality"""
        retrieval_quality = kwargs.get('retrieval_quality', 0.5)
        
        components = RewardComponents(
            correctness=0.0,
            retrieval_penalty=kwargs.get("retrieval_cost", 0.1),
            latency_penalty=kwargs.get("latency_cost", 0.01),
            step_penalty=kwargs.get("step_penalty", 0.05),
            confidence_gain=confidence_gain,
            document_novelty=self._calculate_novelty(kwargs.get("new_docs", [])),
            retrieval_quality=retrieval_quality
        )
        
        # Intermediate rewards focus on information gain and retrieval quality
        reward = (
            components.confidence_gain * 1.0 +
            components.retrieval_quality * 2.0 +  # Higher weight for good retrieval
            components.document_novelty * 0.5 -
            components.retrieval_penalty -
            components.step_penalty
        )
        
        return max(reward, -0.1)
    
    def _calculate_components(self, question: str, answer: str, **kwargs) -> RewardComponents:
        """Calculate individual reward components with retrieval quality"""
        
        # 1. Correctness
        correctness_score = self.answer_evaluator.evaluate(question, answer, kwargs.get('reference_answer'))
        
        # 2. Retrieval penalty (per document)
        retrieval_count = kwargs.get("retrieval_count", 0)
        retrieval_penalty = -retrieval_count * 0.1
        
        # 3. Latency penalty
        latency = kwargs.get("latency", 0.0)
        latency_penalty = -min(latency / 10.0, 1.0)
        
        # 4. Step penalty
        step_count = kwargs.get("step_count", 0)
        step_penalty = -step_count * 0.05
        
        # 5. Retrieval quality (NEW - based on context match)
        retrieved_docs = kwargs.get("retrieved_docs", [])
        relevant_docs = kwargs.get("relevant_docs", [])
        retrieval_quality = self._calculate_retrieval_quality(retrieved_docs, relevant_docs)
        
        # 6. Confidence gain
        confidence_history = kwargs.get("confidence_history", [])
        confidence_gain = self._calculate_confidence_gain(confidence_history)
        
        # 7. Document novelty
        document_novelty = self._calculate_novelty(retrieved_docs)
        
        return RewardComponents(
            correctness=correctness_score,
            retrieval_penalty=retrieval_penalty,
            latency_penalty=latency_penalty,
            step_penalty=step_penalty,
            retrieval_quality=retrieval_quality,
            confidence_gain=confidence_gain,
            document_novelty=document_novelty
        )
    
    def _calculate_novelty(self, docs: List[str]) -> float:
        """Calculate document novelty score"""
        if not docs:
            return 0.0
        
        # Shorter docs = less novelty
        avg_length = sum(len(doc) for doc in docs) / len(docs)
        novelty = min(1.0, avg_length / 500.0)  # Normalize
        return novelty

    def _calculate_confidence_gain(self, confidence_history: List[float]) -> float:
        """Calculate confidence gain from history"""
        if not confidence_history or len(confidence_history) < 2:
            return 0.0
        return confidence_history[-1] - confidence_history[0]
    
    def _calculate_retrieval_quality(self, retrieved_docs: List[str], 
                                     relevant_docs: List[str]) -> float:
        """Calculate how well retrieved docs match relevant docs"""
        if not retrieved_docs or not relevant_docs or self.embedding_model is None:
            return 0.5  # Fallback – Neutral if no ground truth
        
        # Simple semantic similarity using embeddings
        try:
            # Encode documents using pre-loaded model
            retrieved_embeddings = self.embedding_model.encode(retrieved_docs, convert_to_tensor=True)
            relevant_embeddings = self.embedding_model.encode(relevant_docs, convert_to_tensor=True)
            
            # Compute cosine similarities
            similarities = util.cos_sim(retrieved_embeddings, relevant_embeddings)
            
            # For each retrieved doc, find max similarity to any relevant doc
            max_similarities, _ = similarities.max(dim=1)
            
            # Average similarity (higher = better retrieval)
            avg_similarity = max_similarities.mean().item()
            
            # Normalize to [0, 1] range
            quality = max(0.0, min(1.0, avg_similarity))
            
            return quality
            
        except Exception as e:
            # Fallback to simple word overlap
            retrieved_text = " ".join(retrieved_docs).lower()
            relevant_text = " ".join(relevant_docs).lower()
            
            retrieved_words = set(retrieved_text.split()[:100])  # Limit to first 100 words
            relevant_words = set(relevant_text.split()[:100])
            
            if not relevant_words:
                return 0.5
            
            overlap = len(retrieved_words.intersection(relevant_words))
            jaccard = overlap / len(retrieved_words.union(relevant_words))
            
            return jaccard

class ExactMatchEvaluator:
    """Simple exact match evaluator"""
    
    def evaluate(self, question: str, answer: str, reference: str = None) -> float:
        if reference is None:
            # For training, we need ground truth
            return 0.0
        
        # Simple exact match
        answer_lower = answer.lower().strip()
        reference_lower = reference.lower().strip()
        
        if answer_lower == reference_lower:
            return 1.0
        elif reference_lower in answer_lower:
            return 0.5
        else:
            return 0.0