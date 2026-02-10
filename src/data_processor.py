"""
Data processor for loading and preprocessing datasets for adaptive RAG.
"""

import json
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from pathlib import Path
import random
from collections import defaultdict
import logging
import tarfile
import gzip
from tqdm import tqdm

from sentence_transformers import SentenceTransformer


@dataclass
class DataPoint:
    """Single data point for training/evaluation"""
    question: str
    answer: str
    context: Optional[List[str]] = None  # Relevant documents
    difficulty: Optional[str] = None  # simple, medium, hard
    metadata: Optional[Dict] = None  # Additional metadata
    id: Optional[str] = None


@dataclass
class CorpusDocument:
    """Document in the retrieval corpus"""
    id: str
    text: str
    title: Optional[str] = None
    metadata: Optional[Dict] = None
    embedding: Optional[np.ndarray] = None


class DatasetLoader:
    """Load and process datasets for adaptive RAG"""
    
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Initialize paths
        self.data_dir = Path(config.data_path)
        self.corpus_dir = Path(config.corpus_path)
        
        # Create directories if they don't exist
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.corpus_dir.mkdir(parents=True, exist_ok=True)
        
        # Load corpus
        self.corpus = self._load_corpus()
        
        # Embedding model for question difficulty estimation
        self.embedding_model = None
        if config.get("use_embeddings_for_difficulty", True):
            self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    
    def _load_corpus(self) -> List[CorpusDocument]:
        """Load the retrieval corpus"""
        corpus_path = self.corpus_dir / "corpus.jsonl"
        
        if not corpus_path.exists():
            self.logger.warning(f"Corpus not found at {corpus_path}")
            return []
        
        corpus = []
        with open(corpus_path, "r", encoding="utf-8") as f:
            for line in tqdm(f, desc="Loading corpus"):
                data = json.loads(line)
                corpus.append(CorpusDocument(
                    id=data.get("id", str(len(corpus))),
                    text=data.get("text", ""),
                    title=data.get("title", ""),
                    metadata=data.get("metadata", {})
                ))
        
        self.logger.info(f"Loaded {len(corpus)} documents from corpus")
        return corpus
    
    def load_train(self) -> List[DataPoint]:
        """Load training data"""
        return self._load_dataset("train")
    
    def load_val(self) -> List[DataPoint]:
        """Load validation data"""
        return self._load_dataset("val")
    
    def load_test(self) -> List[DataPoint]:
        """Load test data"""
        return self._load_dataset("test")
    
    def load_warmup(self) -> List[DataPoint]:
        """Load warmup/supervised data with CONFIDENCE targets"""
        warmup_path = self.data_dir / "warmup.jsonl"
        
        if warmup_path.exists():
            warmup_data = []
            with open(warmup_path, "r", encoding="utf-8") as f:
                for line in tqdm(f, desc="Loading warmup data"):
                    data = json.loads(line)
                    
                    # Set OUTCOME-based metadata
                    metadata = data.get("metadata", {})
                    
                    # Target confidence
                    if "target_confidence" not in metadata:
                        # Heuristic: longer/complex answers need higher confidence
                        answer_len = len(data.get("answer", ""))
                        if answer_len < 50:
                            metadata["target_confidence"] = 0.80
                        elif answer_len < 150:
                            metadata["target_confidence"] = 0.85
                        else:
                            metadata["target_confidence"] = 0.90
                    
                    # Minimum retrievals (safety constraint only)
                    metadata["min_retrievals"] = 1  # Always at least 1
                    metadata["max_retrievals"] = 6  # Hard upper limit
                    
                    warmup_data.append(DataPoint(
                        question=data["question"],
                        answer=data["answer"],
                        context=data.get("context", []),
                        difficulty=data.get("difficulty"),
                        metadata=metadata,
                        id=data.get("id")
                    ))
            
            self.logger.info(f"Loaded {len(warmup_data)} warmup examples")
            return warmup_data
        else:
            # Generate synthetic warmup
            self.logger.info("Generating synthetic warmup data...")
            return self._create_synthetic_warmup_data()
    
    def _load_dataset(self, split: str) -> List[DataPoint]:
        """Load dataset split"""
        # Try multiple possible file formats
        possible_paths = [
            self.data_dir / f"{split}.jsonl",
            self.data_dir / f"{split}.json",
            self.data_dir / f"{split}.csv",
            self.data_dir / f"{split}_data.jsonl"
        ]
        
        for path in possible_paths:
            if path.exists():
                if path.suffix == ".jsonl":
                    return self._load_from_jsonl(path)
                elif path.suffix == ".json":
                    return self._load_from_json(path)
                elif path.suffix == ".csv":
                    return self._load_from_csv(path)
        
        # If no file found, try to download or use synthetic data
        self.logger.warning(f"No {split} data found. Creating synthetic data.")
        return self._create_synthetic_data(split)
    
    def _load_from_jsonl(self, path: Path) -> List[DataPoint]:
        """Load data from JSONL file"""
        data = []
        with open(path, "r", encoding="utf-8") as f:
            for line_num, line in tqdm(enumerate(f), desc=f"Loading {path.name}"):
                try:
                    item = json.loads(line)
                    data_point = self._parse_data_item(item, str(line_num))
                    if data_point:
                        data.append(data_point)
                except json.JSONDecodeError as e:
                    self.logger.warning(f"Error parsing line {line_num} in {path}: {e}")
        
        self.logger.info(f"Loaded {len(data)} examples from {path}")
        return data
    
    def _load_from_json(self, path: Path) -> List[DataPoint]:
        """Load data from JSON file"""
        with open(path, "r", encoding="utf-8") as f:
            items = json.load(f)
        
        data = []
        if isinstance(items, list):
            for i, item in tqdm(enumerate(items), desc=f"Loading {path.name}"):
                data_point = self._parse_data_item(item, str(i))
                if data_point:
                    data.append(data_point)
        elif isinstance(items, dict):
            # Handle dict format like {'questions': [], 'answers': []}
            questions = items.get("questions", [])
            answers = items.get("answers", [])
            contexts = items.get("contexts", [])
            
            for i, (q, a) in enumerate(zip(questions, answers)):
                context = contexts[i] if i < len(contexts) else None
                data.append(DataPoint(
                    question=q,
                    answer=a,
                    context=context,
                    id=str(i)
                ))
        
        self.logger.info(f"Loaded {len(data)} examples from {path}")
        return data
    
    def _load_from_csv(self, path: Path) -> List[DataPoint]:
        """Load data from CSV file"""
        df = pd.read_csv(path)
        data = []
        
        # Try to find question and answer columns
        question_col = None
        answer_col = None
        context_col = None
        
        for col in df.columns:
            col_lower = col.lower()
            if "question" in col_lower:
                question_col = col
            elif "answer" in col_lower:
                answer_col = col
            elif "context" in col_lower or "text" in col_lower:
                context_col = col
        
        if question_col is None or answer_col is None:
            raise ValueError(f"Could not find question/answer columns in {path}")
        
        for idx, row in df.iterrows():
            context = None
            if context_col:
                if isinstance(row[context_col], str):
                    context = [row[context_col]]
                elif isinstance(row[context_col], list):
                    context = row[context_col]
            
            data.append(DataPoint(
                question=str(row[question_col]),
                answer=str(row[answer_col]),
                context=context,
                id=str(idx)
            ))
        
        self.logger.info(f"Loaded {len(data)} examples from {path}")
        return data
    
    def _map_difficulty(self, raw_difficulty: Any) -> Optional[int]:
        """Map raw difficulty to standardized levels"""
        if isinstance(raw_difficulty, str):
            raw_lower = raw_difficulty.lower()
            if raw_lower in ["easy", "simple", "1"]:
                return 1
            elif raw_lower in ["medium", "intermediate", "2"]:
                return 2
            elif raw_lower in ["hard", "difficult", "3"]:
                return 3
        
        return None
    
    def _parse_data_item(self, item: Dict, default_id: str) -> Optional[DataPoint]:
        """Parse data item with context for retrieval"""
        try:
            # Extract essential fields
            question = item.get("question")
            answer = str(item.get("answer"))
            
            if not question or not answer:
                self.logger.warning(f"Missing question or answer in item {default_id}")
                return None
            
            # Extract context
            context = self._extract_context(item)
            if not context:
                self.logger.warning(f"Item {default_id} has no context - retrieval learning will be limited")
            
            # Extract and map difficulty
            raw_difficulty = item.get("difficulty")
            mapped_difficulty = self._map_difficulty(raw_difficulty)
            
            # Estimate difficulty if not provided
            if not mapped_difficulty:
                mapped_difficulty, _ = self.estimate_difficulty(question)
            
            return DataPoint(
                question=str(question),
                answer=str(answer),
                context=context,
                difficulty=mapped_difficulty,
                metadata={"has_context": context is not None},
                id=item.get("id", default_id)
            )
        
        except Exception as e:
            self.logger.error(f"Error parsing item {default_id}: {e}")
            return None
        
    def _extract_context(self, item: Dict) -> Optional[List[str]]:
        """Extract context from various field names"""
        # Try multiple possible field names
        context = None
        
        if "context" in item:
            context = item["context"]
        elif "docs" in item:
            context = item["docs"]
        elif "documents" in item:
            context = item["documents"]
        elif "evidence" in item:
            context = item["evidence"]
        
        if not context:
            return None
        
        # Convert to list of strings
        if isinstance(context, list):
            return [str(doc) for doc in context if str(doc).strip()]
        elif isinstance(context, str):
            return [context.strip()]
        else:
            return [str(context).strip()]
    
    def _create_synthetic_warmup_data(self, num_examples: int = 100) -> List[DataPoint]:
        """Create synthetic warmup data with confidence targets"""
        warmup_data = []
        
        simple_questions = [
            ("What is the capital of France?", "Paris"),
            ("Who wrote Romeo and Juliet?", "William Shakespeare"),
            ("What is 2+2?", "4"),
        ]
        
        for q, a in simple_questions * (num_examples // len(simple_questions)):
            warmup_data.append(DataPoint(
                question=q,
                answer=a,
                context=[f"Context: {a}"],
                difficulty='simple',
                metadata={
                    "target_confidence": 0.85,
                    "min_retrievals": 1,
                    "max_retrievals": 3
                },
                id=f"warmup_{len(warmup_data)}"
            ))
        
        return warmup_data
    
    def _create_synthetic_data(self, split: str, num_examples: int = 1000) -> List[DataPoint]:
        """Create synthetic training data when no dataset is available"""
        self.logger.info(f"Creating {num_examples} synthetic examples for {split}")
        
        synthetic_data = []
        
        # Question templates for different difficulty levels
        templates = {
            "simple": [
                "What is the capital of {country}?",
                "Who wrote {book}?",
                "When was {person} born?",
                "What is the chemical symbol for {element}?"
            ],
            "medium": [
                "Explain the process of {process}.",
                "What are the main causes of {phenomenon}?",
                "Compare and contrast {thing1} and {thing2}.",
                "How does {system} work?",
                "Was the author of {book1} born before the author of {book2}?",
                "Which {category} received more awards, {thing1} or {thing2}?",
                "What is the relationship between {person1} and {person2}?",
                "Based on {fact1} and {fact2}, what can we conclude about {topic}?"
            ],
            "hard": [
                "Analyze the impact of {event} on {field}.",
                "What are the ethical implications of {technology}?",
                "Explain {theory} and its significance in {field}.",
                "How did {historical_event} lead to {outcome}?"
            ]
        }
        
        # Fillers for template variables
        fillers = {
            "country": ["France", "Japan", "Brazil", "Australia", "Egypt"],
            "book": ["Moby Dick", "Pride and Prejudice", "1984", "The Great Gatsby"],
            "person": ["Albert Einstein", "Marie Curie", "Leonardo da Vinci", "Nelson Mandela"],
            "element": ["Oxygen", "Gold", "Iron", "Carbon"],
            "process": ["photosynthesis", "cellular respiration", "machine learning"],
            "phenomenon": ["climate change", "economic inflation", "social inequality"],
            "thing1": ["democracy", "republic", "capitalism", "socialism"],
            "thing2": ["autocracy", "monarchy", "communism", "feudalism"],
            "system": ["the electoral college", "the stock market", "the immune system"],
            "event": ["the Industrial Revolution", "World War II", "the Digital Revolution"],
            "field": ["economics", "sociology", "environmental science"],
            "technology": ["artificial intelligence", "genetic engineering", "surveillance technology"],
            "theory": ["theory of relativity", "theory of evolution", "quantum theory"],
            "historical_event": ["the French Revolution", "the Renaissance", "the Cold War"],
            "outcome": ["modern democracy", "globalization", "the internet"],
            "book1": ["War and Peace", "To Kill a Mockingbird", "The Odyssey"],
            "book2": ["Crime and Punishment", "The Catcher in the Rye", "The Iliad"],
            "category": ["movie", "novel", "scientific discovery"],
            "topic": ["climate policy", "economic development", "social justice"]
        }
        
        for i in tqdm(range(num_examples), desc="Creating synthetic data"):
            # Randomly select difficulty level
            difficulty = random.choice(["simple", "medium", "hard"])
            
            # Select template
            template = random.choice(templates[difficulty])
            
            # Fill template
            question = template
            for var in fillers:
                if f"{{{var}}}" in question:
                    question = question.replace(f"{{{var}}}", random.choice(fillers[var]))
            
            # Generate synthetic answer
            answer = f"This is a synthetic answer for: {question}"
            
            # Create context (relevant documents)
            context = [
                f"Document about {question.split()[0]}",
                f"Additional information related to {question}",
                f"Historical context for understanding {question}"
            ]
            
            synthetic_data.append(DataPoint(
                question=question,
                answer=answer,
                context=context,
                difficulty=difficulty,
                metadata={"synthetic": True, "split": split},
                id=f"synthetic_{split}_{i}"
            ))
        
        return synthetic_data
    
    def _create_warmup_examples(self, data: List[DataPoint]) -> List[DataPoint]:
        """
        Create warmup examples with CONFIDENCE targets (not retrieval counts).
        Adds metadata to guide initial training.
        """
        warmup_examples = []
        
        for example in data:
            # Set confidence target based on answer complexity
            answer_words = len(example.answer.split())
            difficulty = example.difficulty or "medium"
            
            # Target confidence (outcome-based)
            if answer_words < 20:
                target_conf = 0.80
            elif answer_words < 50:
                target_conf = 0.85
            else:
                target_conf = 0.90
            
            # Safety constraints (not targets!)
            min_ret = 1  # Always retrieve at least once
            max_ret = 6  # Never more than 6
            
            example.metadata = example.metadata or {}
            example.metadata.update({
                "target_confidence": target_conf,
                "min_retrievals": min_ret,
                "max_retrievals": max_ret
            })
            
            warmup_examples.append(example)
        
        return warmup_examples
    
    def estimate_difficulty(self, question: str) -> Tuple[str, float]:
        """
        Estimate question difficulty
        
        Returns:
            Tuple of (difficulty_level, confidence_score)
        """
        if not self.embedding_model:
            return ("medium", 0.5)
        
        # Simple heuristic based on question length and complexity
        question_lower = question.lower()
        
        # Check for multi-hop indicators
        is_multi_hop = any(indicator in question_lower for indicator in [
            "based on", "given that", "if", "then", "therefore", "thus",
            "first", "second", "then", "finally", "compare", "contrast"
        ])
        
        # Check for complex reasoning indicators
        is_complex = any(indicator in question_lower for indicator in [
            "explain", "analyze", "evaluate", "discuss", "critique",
            "implications", "significance", "impact", "relationship"
        ])
        
        # Check for simple fact indicators
        is_simple = any(indicator in question_lower for indicator in [
            "what is", "who is", "when was", "where is",
            "define", "list", "name"
        ]) and not is_multi_hop and not is_complex
        
        # Calculate question complexity score
        word_count = len(question.split())
        char_count = len(question)
        avg_word_length = char_count / max(word_count, 1)
        
        complexity_score = (
            (word_count / 50) +  # Normalize word count
            (avg_word_length / 10) +  # Normalize word length
            (1.0 if is_multi_hop else 0.0) +
            (0.7 if is_complex else 0.0) -
            (0.5 if is_simple else 0.0)
        )
        
        # Classify based on complexity score
        if is_multi_hop:
            difficulty = "multi-hop"
            confidence = 0.8
        elif complexity_score > 1.5:
            difficulty = "hard"
            confidence = min(0.9, complexity_score / 3.0)
        elif complexity_score > 0.8:
            difficulty = "medium"
            confidence = 0.7
        else:
            difficulty = "simple"
            confidence = max(0.6, 1.0 - complexity_score)
        
        return difficulty, confidence
    
    def stratify_by_difficulty(self, data: List[DataPoint]) -> Dict[str, List[DataPoint]]:
        """Stratify data by difficulty level"""
        stratified = defaultdict(list)
        
        for dp in data:
            if dp.difficulty:
                stratified[dp.difficulty].append(dp)
            else:
                # Estimate difficulty if not provided
                difficulty, _ = self.estimate_difficulty(dp.question)
                stratified[difficulty].append(dp)
        
        return dict(stratified)
    
    def save_dataset(self, data: List[DataPoint], path: Path):
        """Save dataset to JSONL file"""
        with open(path, "w", encoding="utf-8") as f:
            for dp in data:
                item = {
                    "id": dp.id,
                    "question": dp.question,
                    "answer": dp.answer,
                    "difficulty": dp.difficulty,
                    "metadata": dp.metadata
                }
                if dp.context:
                    item["context"] = dp.context
                
                f.write(json.dumps(item) + '\n')
        
        self.logger.info(f"Saved {len(data)} examples to {path}")