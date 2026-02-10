import gymnasium as gym
import numpy as np
from typing import Dict, Tuple, Optional, Any, List
from dataclasses import dataclass
import time
import random
from abc import ABC, abstractmethod
import logging
logger = logging.getLogger(__name__)

@dataclass
class RetrievalResult:
    documents: list
    scores: list
    query_used: str
    retrieval_time: float

@dataclass
class QuestionSample:
    """Return type for question sampling"""
    question: str
    answer: Optional[str] = None
    context: Optional[List[str]] = None
    difficulty: Optional[str] = None
    metadata: Optional[dict] = None

class RAGEnvironment(gym.Env):
    """Custom environment for adaptive RAG"""
    
    def __init__(self, config, generator, retriever, reward_calculator, data_loader=None):
        """
        Modified __init__ to accept data_loader
        
        Args:
            config: Configuration object
            generator: Text generator model
            retriever: Document retriever
            reward_calculator: Reward calculation module
            data_loader: DatasetLoader instance (optional, for training)
        """
        super().__init__()
        
        self.config = config
        self.generator = generator
        self.retriever = retriever
        self.reward_calculator = reward_calculator
        
        # Store data loader
        self.data_loader = data_loader
        
        # Cache for loaded data (to avoid reloading every episode)
        self._cached_train_data = None
        self._cached_val_data = None
        self._current_split = "train"  # 'train' or 'val'

        # Initialize state encoder FIRST
        from src.state_encoder import AdaptiveRAGStateEncoder
        self.state_encoder = AdaptiveRAGStateEncoder(config)
        
        # Get ACTUAL state dimension from state encoder
        actual_state_dim = self.state_encoder.get_state_dim()
        
        # Define observation space with CORRECT dimension
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(actual_state_dim,), dtype=np.float32
        )
        
        logger.info(f"Environment initialized with state_dim={actual_state_dim}")
        
        # Action space: 0=retrieve_more, 1=re_query, 2=answer
        self.action_space = gym.spaces.Discrete(3)
        
        # Initialize episode state
        # self.reset()
        
        # Track ground truth for evaluation
        self.relevant_docs = []
        self.current_relevant_docs = []
        self._reference_answer = None
        self._question_id = None
    
    def reset(self, question: Optional[str] = None, 
              relevant_docs: Optional[List[str]] = None, **kwargs):
        """
        Reset environment for new question.
        
        Args:
            question: Optional question text. If None, samples from dataset.
            relevant_docs: Optional ground truth relevant documents.
            **kwargs: Additional arguments
        
        Returns:
            Initial state observation
        """
        # Sample question if not provided
        if question is None:
            self.question = self._sample_question()
        else:
            self.question = question
            
            # Use provided relevant_docs or what's already stored
            if relevant_docs is not None:
                self.relevant_docs = relevant_docs
                self.current_relevant_docs = relevant_docs.copy()
            
            # If reference answer provided in kwargs, use it
            if "reference_answer" in kwargs:
                self._reference_answer = kwargs["reference_answer"]
        
        # Reset episode state
        self.retrieved_docs = []
        self.all_retrieval_results = []
        self.partial_answer = ""
        self.step_count = 0
        self.query_reformulations = 0
        self.confidence_history = []
        self.start_time = time.time()
        
        # Initial retrieval
        initial_result = self._retrieve(self.question)
        self.retrieved_docs.extend(initial_result.documents)
        self.all_retrieval_results.append(initial_result)
        
        # Generate initial partial answer
        self.partial_answer, self.current_confidence = self._generate_partial_answer()
        self.confidence_history.append(self.current_confidence)
        
        return self._get_state()
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        """Execute action and return next state, reward, done, info"""
        
        self.step_count += 1
        info = {"action": None, "confidence": self.current_confidence}
        
        # Store initial confidence for reward calculation
        if not hasattr(self, "initial_confidence") or self.step_count == 1:
            self.initial_confidence = self.current_confidence
        
        # Action 0: Retrieve more documents
        if action == 0:
            result = self._retrieve_more()
            new_docs = result.documents
            info["action"] = "retrieve_more"
            info["new_docs"] = new_docs
            info["retrieval_time"] = result.retrieval_time
            
            # Calculate retrieval quality
            info["retrieval_quality"] = self._compute_retrieval_quality(new_docs)
            
            # Update partial answer and confidence
            self.partial_answer, self.current_confidence = self._generate_partial_answer()
            self.confidence_history.append(self.current_confidence)
            
            # Intermediate reward
            confidence_gain = self._get_confidence_gain()
            reward = self.reward_calculator.calculate_intermediate(
                confidence_gain,
                retrieval_quality=info["retrieval_quality"],
                retrieval_cost=0.1,
                step_penalty=0.05
            )
            
            done = self._should_terminate()
        
        # Action 1: Re-query with reformulation
        elif action == 1:
            result = self._re_query()
            new_docs = result.documents
            info["action"] = "re_query"
            info["new_docs"] = new_docs
            info["retrieval_time"] = result.retrieval_time
            
            info["retrieval_quality"] = self._compute_retrieval_quality(new_docs)
            
            self.partial_answer, self.current_confidence = self._generate_partial_answer()
            self.confidence_history.append(self.current_confidence)
            
            confidence_gain = self._get_confidence_gain()
            reward = self.reward_calculator.calculate_intermediate(
                confidence_gain,
                retrieval_quality=info["retrieval_quality"],
                retrieval_cost=0.15,  # Slightly higher cost for re-query
                step_penalty=0.05
            )
            
            done = self._should_terminate()
        
        # Action 2: Generate final answer
        elif action == 2:
            final_answer = self._generate_final_answer()
            info["action"] = "answer"
            info["final_answer"] = final_answer
            info["confidence"] = self.current_confidence
            
            # Final reward with ALL context
            reward = self.reward_calculator.calculate(
                question=self.question,
                answer=final_answer,
                reference_answer=self._reference_answer,
                retrieval_count=len(self.all_retrieval_results),
                step_count=self.step_count,
                latency=time.time() - self.start_time,
                confidence_gain=self._get_confidence_gain(),
                initial_confidence=self.initial_confidence,
                final_confidence=self.current_confidence,
                confidence_history=self.confidence_history,
                retrieved_docs=self.retrieved_docs,
                relevant_docs=self.relevant_docs
            )
            
            done = True
        
        next_state = self._get_state()
        return next_state, reward, done, info
    
    def _compute_retrieval_quality(self, retrieved_docs: List[str]) -> float:
        """Compute how well retrieved docs match relevant docs"""
        if not retrieved_docs or not self.relevant_docs:
            return 0.5  # Neutral if no ground truth
        
        # Simple overlap-based quality
        retrieved_text = ' '.join(retrieved_docs).lower()
        relevant_text = ' '.join(self.relevant_docs).lower()
        
        # Count overlapping words
        retrieved_words = set(retrieved_text.split())
        relevant_words = set(relevant_text.split())
        
        if not relevant_words:
            return 0.5
        
        overlap = len(retrieved_words.intersection(relevant_words))
        total_relevant = len(relevant_words)
        
        # Precision-like metric
        quality = min(1.0, overlap / max(10, total_relevant * 0.5))
        return quality
    
    def _generate_final_answer(self) -> str:
        """Generate final answer using retrieved docs"""
        if not self.retrieved_docs:
            return self.generator.generate(f"Question: {self.question}\nAnswer:")
        
        # Use retrieved docs to generate answer
        context = "\n".join([f"Document {i+1}: {doc}" for i, doc in enumerate(self.retrieved_docs[:5])])
        prompt = f"""Based on the following documents, answer the question.
        
        Documents:
        {context}
        
        Question: {self.question}
        
        Answer: """
        
        return self.generator.generate(prompt)
    
    def _retrieve_more(self) -> RetrievalResult:
        """Retrieve additional documents"""
        # Use same query but get more results
        result = self._retrieve(self.question, offset=len(self.retrieved_docs))
        self.retrieved_docs.extend(result.documents)
        self.all_retrieval_results.append(result)
        return result
    
    def _re_query(self) -> RetrievalResult:
        """Reformulate query and retrieve"""
        if self.query_reformulations >= self.config.environment.max_query_reformulations:
            # Fall back to retrieve_more if max reformulations reached
            return self._retrieve_more()
        
        # Generate query reformulation
        reformulated_query = self._generate_query_reformulation()
        self.query_reformulations += 1
        
        result = self._retrieve(reformulated_query)
        self.retrieved_docs.extend(result.documents)
        self.all_retrieval_results.append(result)
        return result
    
    def _generate_query_reformulation(self) -> str:
        """Generate a reformulated query based on current context"""
        prompt = f"""
        Original query: {self.question}
        Current partial answer: {self.partial_answer}
        Retrieved documents so far: {len(self.retrieved_docs)}
        
        Generate a better search query to find missing information.
        Focus on aspects not covered in current documents.
        Query: """
        
        return self.generator.generate(prompt, max_tokens=50)
    
    def _get_state(self) -> np.ndarray:
        """Encode current state as numpy array"""
        return self.state_encoder.encode(
            question=self.question,
            partial_answer=self.partial_answer,
            confidence=self.current_confidence,
            retrieved_docs=self.retrieved_docs,
            step_count=self.step_count,
            query_reformulations=self.query_reformulations,
            confidence_history=self.confidence_history,
            question_difficulty=getattr(self, "_question_difficulty", None)
        )
    
    def _should_terminate(self) -> bool:
        """Check if we should terminate the episode"""
        if self.step_count >= self.config.environment.max_retrieval_steps:
            return True
        
        if len(self.confidence_history) >= 3:
            # Check if confidence has plateaued or decreasing
            recent_confidences = self.confidence_history[-3:]
            if np.std(recent_confidences) < 0.05:  # Plateau
                return True
        
        if self.current_confidence >= self.config.environment.confidence_threshold:
            # High confidence, might be ready to answer
            return True
        
        return False
    
    def _retrieve(self, query: str, offset: int = 0) -> RetrievalResult:
        start_time = time.time()
        results = self.retriever.retrieve(query, top_k=self.config.environment.retrieval_batch_size, offset=offset)
        
        documents = [r["text"] for r in results]
        scores = [r["score"] for r in results]
        
        return RetrievalResult(
            documents=documents,
            scores=scores,
            query_used=query,
            retrieval_time=time.time() - start_time
        )

    def _generate_partial_answer(self) -> Tuple[str, float]:
        if not self.retrieved_docs:
            prompt = f"Question: {self.question}\nProvide a brief answer:"
        else:
            context = "\n".join(self.retrieved_docs[:3])
            prompt = f"Context: {context}\n\nQuestion: {self.question}\nProvide a brief answer:"
        
        answer, confidence = self.generator.generate_with_confidence(prompt, max_tokens=100)
        return answer, confidence
    
    def _get_confidence_gain(self) -> float:
        if len(self.confidence_history) < 2:
            return 0.0
        return self.confidence_history[-1] - self.confidence_history[0]
    
    def set_mode(self, mode: str):
        """
        Set environment mode: 'train' or 'val'
        
        Args:
            mode: Either 'train' for training or 'val' for validation/evaluation
        """
        if mode not in ["train", "val", "test"]:
            raise ValueError(f"Mode must be 'train', 'val', or 'test', got {mode}")
        
        self._current_split = mode
        logger.info(f"Environment mode set to: {mode}")
    
    
    def _sample_question(self) -> str:
        """
        Sample a question from the appropriate dataset split.
        Uses train data during training, val data during validation.
        
        Returns:
            str: Sampled question text
        """
        # Check if data loader is available
        if self.data_loader is None:
            logger.warning("No data_loader provided. Using fallback synthetic question.")
            return self._fallback_question()
        
        # Load appropriate dataset based on current mode
        try:
            if self._current_split == "train":
                data = self._get_or_load_train_data()
            elif self._current_split == "val":
                data = self._get_or_load_val_data()
            elif self._current_split == "test":
                data = self._get_or_load_test_data()
            else:
                logger.warning(f"Unknown split: {self._current_split}, defaulting to train")
                data = self._get_or_load_train_data()
            
            if not data:
                logger.warning(f"No data available for split '{self._current_split}'. Using fallback.")
                return self._fallback_question()
            
            # Sample random data point
            data_point = random.choice(data)
            
            # Store metadata for later use (reward calculation, evaluation)
            self._store_sample_metadata(data_point)
            
            return data_point.question
            
        except Exception as e:
            logger.error(f"Error sampling question: {e}")
            return self._fallback_question()
    
    
    def _get_or_load_train_data(self) -> List:
        """Load and cache training data"""
        if self._cached_train_data is None:
            logger.info("Loading training data...")
            self._cached_train_data = self.data_loader.load_train()
            logger.info(f"Loaded {len(self._cached_train_data)} training examples")
        return self._cached_train_data
    
    
    def _get_or_load_val_data(self) -> List:
        """Load and cache validation data"""
        if self._cached_val_data is None:
            logger.info("Loading validation data...")
            self._cached_val_data = self.data_loader.load_val()
            logger.info(f"Loaded {len(self._cached_val_data)} validation examples")
        return self._cached_val_data
    
    
    def _get_or_load_test_data(self) -> List:
        """Load test data (not cached since typically used once)"""
        logger.info("Loading test data...")
        test_data = self.data_loader.load_test()
        logger.info(f"Loaded {len(test_data)} test examples")
        return test_data
    
    
    def _store_sample_metadata(self, data_point):
        """
        Store metadata from sampled data point for later use.
        
        Args:
            data_point: DataPoint instance from DatasetLoader
        """
        # Store reference answer for reward calculation
        self._reference_answer = data_point.answer
        
        # Store relevant documents (ground truth context) for retrieval quality
        self.relevant_docs = data_point.context if data_point.context else []
        self.current_relevant_docs = self.relevant_docs.copy()
        
        # Store question ID for tracking
        self._question_id = data_point.id
        
        # Store difficulty for analysis
        self._question_difficulty = data_point.difficulty
        
        # Store any additional metadata
        self._question_metadata = data_point.metadata or {}
        
        logger.debug(f"Sampled question {self._question_id} "
                    f"(difficulty: {self._question_difficulty}, "
                    f"has_context: {len(self.relevant_docs) > 0})")
    
    
    def _fallback_question(self) -> str:
        """
        Fallback question when data loader is unavailable or fails.
        Returns a simple synthetic question.
        """
        fallback_questions = [
            "What is the capital of France?",
            "Who invented the telephone?",
            "When was the United Nations founded?",
            "What is the largest planet in our solar system?",
            "Who wrote Romeo and Juliet?"
        ]
        
        question = random.choice(fallback_questions)
        
        # Set default metadata
        self._reference_answer = "This is a fallback answer."
        self.relevant_docs = []
        self._question_id = f"fallback_{hash(question)}"
        self._question_difficulty = "simple"
        self._question_metadata = {"fallback": True}
        
        logger.warning(f"Using fallback question: {question}")
        return question
    
    
    def get_current_reference_answer(self) -> Optional[str]:
        """Get the reference answer for the current question"""
        return self._reference_answer
    
    
    def get_current_question_id(self) -> Optional[str]:
        """Get the ID of the current question"""
        return self._question_id
    
    
    def get_current_difficulty(self) -> Optional[str]:
        """Get the difficulty of the current question"""
        return self._question_difficulty