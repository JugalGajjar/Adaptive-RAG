"""
Evaluation script for Adaptive Retrieval RAG
"""

import os
import hydra
from omegaconf import DictConfig, OmegaConf
import torch
import pandas as pd
import numpy as np
from pathlib import Path
import json
from typing import Dict, List
import time
import logging
from collections import defaultdict
from tqdm import tqdm

from src.environment import RAGEnvironment
from src.policy_network import AdaptivePolicyNetwork
from src.agent import AdaptiveRAGAgent
from src.reward_calculator import AdaptiveRAGRewardCalculator
from src.data_processor import DatasetLoader
from src.utils.model_initializers import initialize_generator, initialize_retriever, get_best_device

logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../config", config_name="config")
def main(cfg: DictConfig):
    print("Loading evaluation configuration...")
    
    # Load data
    data_loader = DatasetLoader(cfg.data)
    test_data = data_loader.load_test()
    
    print(f"Loaded {len(test_data)} test examples")
    
    # Load trained agent
    print("Loading trained agent...")
    checkpoint_path = Path(cfg.evaluation.checkpoint_path)
    os.makedirs(checkpoint_path.parent, exist_ok=True)
    agent = load_trained_agent(checkpoint_path, cfg, data_loader)
    
    # Set environment to test mode
    agent.env.set_mode("test")  # Will sample from test split
    
    # Run evaluation
    print(f"Evaluating on {min(len(test_data), cfg.evaluation.max_samples)} questions...")
    results = run_evaluation(agent, test_data, cfg)
    
    # Compute metrics
    print("Computing metrics...")
    metrics = compute_all_metrics(results, cfg)
    
    # Save results
    output_dir = Path(cfg.evaluation.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save detailed results
    results_df = pd.DataFrame(results)
    results_path = output_dir / "detailed_results.csv"
    results_df.to_csv(results_path, index=False)
    print(f"Detailed results saved to {results_path}")
    
    # Save summary metrics
    metrics_path = output_dir / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved to {metrics_path}")
    
    # Print summary
    print_evaluation_summary(metrics)
    
    return metrics


def run_evaluation(agent, test_data, cfg) -> List[Dict]:
    """
    Run evaluation on test data.
    """
    results = []
    max_samples = min(len(test_data), cfg.evaluation.max_samples)
    
    for i in tqdm(range(max_samples), desc="Running Evaluation"):
        # Get the specific test item for reference
        item = test_data[i]
        
        # Record start time
        start_time = time.time()

        state = agent.env.reset(
            question=item.question,
            relevant_docs=item.context if item.context else None
        )
        
        # Initialize episode tracking
        episode_info = {
            "question_id": agent.env.get_current_question_id(),
            "question": agent.env.question,
            "difficulty": agent.env.get_current_difficulty(),
            "reference_answer": agent.env.get_current_reference_answer(),
            "context_provided": len(agent.env.relevant_docs) > 0,
            "actions": [],
            "confidences": [],
            "retrieved_docs_count": [],
            "retrieval_qualities": [],
            "latency": 0.0
        }
        
        # Run episode
        while True:
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(agent.device)
            action, _ = agent.policy_net.get_action(
                state_tensor, 
                epsilon=0.0,  # No exploration during evaluation
                training=False
            )
            
            next_state, reward, done, info = agent.env.step(action)
            
            # Record step information
            episode_info["actions"].append(info.get("action", "unknown"))
            episode_info["confidences"].append(agent.env.current_confidence)
            episode_info["retrieved_docs_count"].append(len(agent.env.retrieved_docs))
            
            if "retrieval_quality" in info:
                episode_info["retrieval_qualities"].append(info["retrieval_quality"])
            
            state = next_state
            
            if done:
                # Record final information
                episode_info["final_answer"] = info.get("final_answer", "")
                episode_info["final_confidence"] = info.get("confidence", 0.0)
                episode_info["latency"] = time.time() - start_time
                
                # Check correctness
                reference = episode_info["reference_answer"]
                if reference:
                    episode_info["correct"] = check_correctness(
                        episode_info["final_answer"], 
                        reference
                    )
                else:
                    episode_info["correct"] = None
                
                # Compute retrieval metrics if context was provided
                if episode_info["context_provided"]:
                    episode_info["retrieval_metrics"] = compute_retrieval_metrics(
                        agent.env.retrieved_docs, 
                        agent.env.relevant_docs
                    )
                
                break
        
        # Compute episode-level metrics
        episode_metrics = compute_episode_metrics(episode_info)
        results.append({**episode_info, **episode_metrics})
        
        # Progress update
        if (i + 1) % 10 == 0:
            print(f"Evaluated {i + 1}/{max_samples} questions...")
    
    return results


def compute_retrieval_metrics(retrieved_docs, relevant_docs):
    """Compute retrieval precision, recall, F1"""
    if not retrieved_docs or not relevant_docs:
        return {
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "overlap_words": 0,
            "total_relevant_words": 0
        }
    
    # Simple word overlap metrics
    retrieved_text = " ".join(retrieved_docs).lower()
    relevant_text = " ".join(relevant_docs).lower()
    
    retrieved_words = set(retrieved_text.split())
    relevant_words = set(relevant_text.split())
    
    if not relevant_words:
        return {
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "overlap_words": 0,
            "total_relevant_words": 0
        }
    
    # Basic metrics
    intersection = retrieved_words.intersection(relevant_words)
    
    precision = len(intersection) / len(retrieved_words) if retrieved_words else 0
    recall = len(intersection) / len(relevant_words)
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "overlap_words": len(intersection),
        "total_relevant_words": len(relevant_words)
    }


def compute_episode_metrics(episode_info: Dict) -> Dict:
    """Compute metrics for a single episode"""
    metrics = {
        "retrieval_count": episode_info["actions"].count("retrieve_more") + 
                          episode_info["actions"].count("re_query"),
        "query_reformulations": episode_info["actions"].count("re_query"),
        "total_steps": len(episode_info["actions"]),
        "max_confidence": max(episode_info["confidences"]) if episode_info["confidences"] else 0,
        "final_confidence": episode_info.get("final_confidence", 0),
        "latency": episode_info.get("latency", 0)
    }
    
    # Compute confidence gain
    if len(episode_info["confidences"]) >= 2:
        metrics["confidence_gain"] = episode_info["confidences"][-1] - episode_info["confidences"][0]
    else:
        metrics["confidence_gain"] = 0
    
    # Average retrieval quality
    if episode_info["retrieval_qualities"]:
        metrics["avg_retrieval_quality"] = np.mean(episode_info["retrieval_qualities"])
    else:
        metrics["avg_retrieval_quality"] = 0.5
    
    return metrics


def compute_all_metrics(results: List[Dict], cfg) -> Dict:
    """Compute aggregate metrics across all results"""
    if not results:
        return {}
    
    # Overall metrics
    metrics = {
        "total_questions": len(results),
        "accuracy": np.mean([r.get("correct", False) for r in results if r.get("correct") is not None]),
        "avg_retrieval_count": np.mean([r["retrieval_count"] for r in results]),
        "avg_latency": np.mean([r["latency"] for r in results]),
        "avg_confidence_gain": np.mean([r["confidence_gain"] for r in results]),
        "avg_retrieval_quality": np.mean([r.get("avg_retrieval_quality", 0.5) for r in results])
    }
    
    # Energy efficiency (Higher is better)
    baseline_retrievals = 5  # Fixed retrieval baseline
    baseline_latency = 10.0  # Fixed latency baseline
    
    retrieval_efficiency = baseline_retrievals / max(metrics["avg_retrieval_count"], 0.1)
    latency_efficiency = baseline_latency / max(metrics["avg_latency"], 0.1)
    metrics["energy_efficiency"] = (retrieval_efficiency + latency_efficiency) / 2
    
    # Adaptive score: balance of accuracy and efficiency
    metrics["adaptive_score"] = (
        metrics["accuracy"] * 0.5 +
        min(retrieval_efficiency, 2.0) * 0.25 +  # Cap at 2x improvement
        min(latency_efficiency, 2.0) * 0.25
    )
    
    # Per-difficulty metrics
    difficulty_groups = defaultdict(list)
    for r in results:
        difficulty = r.get("difficulty", "unknown")
        difficulty_groups[difficulty].append(r)
    
    metrics["difficulty_metrics"] = {}
    for difficulty, group in difficulty_groups.items():
        metrics["difficulty_metrics"][difficulty] = {
            "count": len(group),
            "accuracy": np.mean([r.get("correct", False) for r in group if r.get("correct") is not None]),
            "avg_retrieval_count": np.mean([r["retrieval_count"] for r in group]),
            "avg_latency": np.mean([r["latency"] for r in group]),
            "avg_confidence_gain": np.mean([r["confidence_gain"] for r in group])
        }
    
    # Retrieval metrics (only for questions with context)
    with_context = [r for r in results if r.get("retrieval_metrics")]
    if with_context:
        metrics["retrieval_precision"] = np.mean([r["retrieval_metrics"]["precision"] for r in with_context])
        metrics["retrieval_recall"] = np.mean([r["retrieval_metrics"]["recall"] for r in with_context])
        metrics["retrieval_f1"] = np.mean([r["retrieval_metrics"]["f1"] for r in with_context])
    
    return metrics


def check_correctness(answer: str, reference: str) -> bool:
    """Check if answer is correct"""
    if not answer or not reference:
        return False
    
    answer_lower = answer.lower().strip()
    reference_lower = reference.lower().strip()
    
    return answer_lower == reference_lower or reference_lower in answer_lower


def print_evaluation_summary(metrics: Dict):
    """Print formatted evaluation summary"""
    print("\n" + "="*60)
    print("EVALUATION SUMMARY")
    print("="*60)
    
    print(f"\nOverall Performance:")
    print(f"  Total Questions:        {metrics.get('total_questions', 0)}")
    print(f"  Accuracy:               {metrics.get('accuracy', 0):.3f}")
    print(f"  Avg Retrieval Count:    {metrics.get('avg_retrieval_count', 0):.2f}")
    print(f"  Avg Latency:            {metrics.get('avg_latency', 0):.3f}s")
    print(f"  Energy Efficiency:      {metrics.get('energy_efficiency', 0):.3f}")
    print(f"  Adaptive Score:         {metrics.get('adaptive_score', 0):.3f}")
    
    if "retrieval_precision" in metrics:
        print(f"\nRetrieval Quality:")
        print(f"  Precision:              {metrics.get('retrieval_precision', 0):.3f}")
        print(f"  Recall:                 {metrics.get('retrieval_recall', 0):.3f}")
        print(f"  F1:                     {metrics.get('retrieval_f1', 0):.3f}")
    
    if "difficulty_metrics" in metrics:
        print(f"\nPer-Difficulty Metrics:")
        for difficulty, diff_metrics in sorted(metrics["difficulty_metrics"].items()):
            print(f"  {difficulty.capitalize()}:")
            print(f"    Count:              {diff_metrics['count']}")
            print(f"    Accuracy:           {diff_metrics['accuracy']:.3f}")
            print(f"    Retrievals:         {diff_metrics['avg_retrieval_count']:.2f}")
            print(f"    Latency:            {diff_metrics['avg_latency']:.3f}s")
    
    print("="*60)


def load_trained_agent(checkpoint_path: Path, cfg: DictConfig, data_loader: DatasetLoader):
    """Load trained agent from checkpoint"""
    # Initialize components
    generator = initialize_generator(cfg.model.generator)
    retriever = initialize_retriever(cfg.model.retriever, data_loader.corpus)
    reward_calculator = AdaptiveRAGRewardCalculator(cfg)
    
    # Initialize environment WITH data_loader
    env = RAGEnvironment(
        cfg, 
        generator, 
        retriever, 
        reward_calculator,
        data_loader=data_loader
    )
    
    # Initialize policy network
    state_dim = env.observation_space.shape[0]
    policy_net = AdaptivePolicyNetwork(cfg, state_dim)
    
    # Initialize agent
    device = get_best_device()
    agent = AdaptiveRAGAgent(cfg, policy_net, env, device)
    
    # Load checkpoint
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    agent.policy_net.load_state_dict(checkpoint["policy_state_dict"])
    agent.target_net.load_state_dict(checkpoint["target_state_dict"])
    
    print(f"Loaded checkpoint from {checkpoint_path}")
    print(f"Checkpoint step count: {checkpoint.get('step_count', 'unknown')}")
    
    return agent


if __name__ == "__main__":
    main()