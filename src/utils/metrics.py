"""
Evaluation metrics for Adaptive Retrieval RAG system.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
from collections import defaultdict
import json
import time
from sklearn.metrics import precision_score, recall_score, f1_score
import torch
import pandas as pd


@dataclass
class MetricResult:
    """Container for metric results"""
    name: str
    value: float
    description: str
    metadata: Optional[Dict] = None


@dataclass
class EvaluationMetrics:
    """Collection of evaluation metrics"""
    accuracy: float
    avg_retrieval_count: float
    avg_latency: float
    energy_efficiency: float
    adaptive_score: float
    per_difficulty_metrics: Dict[str, Dict[str, float]]
    details: Dict[str, Any]


class RAGMetrics:
    """Compute metrics for RAG systems"""
    
    @staticmethod
    def compute_all_metrics(results: List[Dict], config: Dict) -> EvaluationMetrics:
        """Compute all evaluation metrics"""
        
        # Basic metrics
        accuracy = RAGMetrics.compute_accuracy(results)
        avg_retrieval_count = RAGMetrics.compute_avg_retrieval_count(results)
        avg_latency = RAGMetrics.compute_avg_latency(results)
        
        # Efficiency metrics
        energy_efficiency = RAGMetrics.compute_energy_efficiency(
            results, config.get("energy_config", {})
        )
        
        # Adaptive behavior metrics
        adaptive_score = RAGMetrics.compute_adaptive_score(results, config)
        
        # Per-difficulty metrics
        per_difficulty_metrics = RAGMetrics.compute_per_difficulty_metrics(results)
        
        # Additional detailed metrics
        details = RAGMetrics.compute_detailed_metrics(results, config)
        
        return EvaluationMetrics(
            accuracy=accuracy,
            avg_retrieval_count=avg_retrieval_count,
            avg_latency=avg_latency,
            energy_efficiency=energy_efficiency,
            adaptive_score=adaptive_score,
            per_difficulty_metrics=per_difficulty_metrics,
            details=details
        )
    
    @staticmethod
    def compute_accuracy(results: List[Dict]) -> float:
        """Compute answer accuracy"""
        if not results:
            return 0.0
        
        correct = 0
        total = 0
        
        for result in results:
            if "correct" in result:
                correct += 1 if result["correct"] else 0
                total += 1
            elif "reference_answer" in result and "final_answer" in result:
                # Compute exact match
                if result["final_answer"].strip().lower() == result["reference_answer"].strip().lower():
                    correct += 1
                total += 1
        
        return correct / total if total > 0 else 0.0
    
    @staticmethod
    def compute_avg_retrieval_count(results: List[Dict]) -> float:
        """Compute average number of retrieval actions per question"""
        if not results:
            return 0.0
        
        total_retrievals = 0
        for result in results:
            total_retrievals += result.get("retrieval_count", 0)
        
        return total_retrievals / len(results)
    
    @staticmethod
    def compute_avg_latency(results: List[Dict]) -> float:
        """Compute average latency per question (in seconds)"""
        if not results:
            return 0.0
        
        total_latency = 0.0
        valid_results = 0
        
        for result in results:
            latency = result.get("latency", 0.0)
            if latency > 0:
                total_latency += latency
                valid_results += 1
        
        return total_latency / valid_results if valid_results > 0 else 0.0
    
    @staticmethod
    def compute_energy_efficiency(results: List[Dict], energy_config: Dict) -> float:
        """
        Compute energy efficiency score.
        
        Energy model assumptions:
        - Each retrieval: ~1 energy unit
        - Each generation step: ~0.5 energy units
        - Each re-query: ~0.3 energy units
        """
        if not results:
            return 0.0
        
        retrieval_weight = energy_config.get("retrieval_weight", 1.0)
        generation_weight = energy_config.get("generation_weight", 0.5)
        requery_weight = energy_config.get("requery_weight", 0.3)
        
        total_energy = 0
        total_optimal_energy = 0
        
        for result in results:
            # Compute actual energy
            retrieval_count = result.get("retrieval_count", 0)
            total_steps = result.get("total_steps", 0)
            query_reformulations = result.get("query_reformulations", 0)
            
            actual_energy = (
                retrieval_count * retrieval_weight +
                total_steps * generation_weight +
                query_reformulations * requery_weight
            )
            total_energy += actual_energy
            
            # Compute optimal energy (based on difficulty)
            difficulty = result.get("difficulty", "medium")
            optimal_counts = {
                "simple": {"retrievals": 1, "steps": 2},
                "medium": {"retrievals": 2, "steps": 3},
                "complex": {"retrievals": 3, "steps": 4},
                "multi-hop": {"retrievals": 4, "steps": 5}
            }
            optimal = optimal_counts.get(difficulty, optimal_counts["medium"])
            
            optimal_energy = (
                optimal["retrievals"] * retrieval_weight +
                optimal["steps"] * generation_weight
            )
            total_optimal_energy += optimal_energy
        
        # Efficiency = optimal energy / actual energy
        if total_energy == 0:
            return 0.0
        
        efficiency = total_optimal_energy / total_energy
        return min(2.0, max(0.0, efficiency))  # Clip to [0, 2]
    
    @staticmethod
    def compute_adaptive_score(results: List[Dict], config: Dict) -> float:
        """
        Compute adaptive behavior score.
        
        Measures how well the system adapts retrieval behavior to question difficulty.
        """
        if not results:
            return 0.0
        
        # Group by difficulty
        difficulty_groups = defaultdict(list)
        for result in results:
            difficulty = result.get("difficulty", "unknown")
            difficulty_groups[difficulty].append(result)
        
        scores = []
        
        for difficulty, group_results in difficulty_groups.items():
            if difficulty == "unknown" or len(group_results) < 3:
                continue
            
            # Compute average retrievals for this difficulty
            avg_retrievals = np.mean([r.get("retrieval_count", 0) for r in group_results])
            
            # Expected retrievals based on difficulty
            expected_retrievals = {
                "simple": 1.0,
                "medium": 2.0,
                "complex": 3.0,
                "multi-hop": 4.0
            }.get(difficulty, 2.0)
            
            # Score based on closeness to expected (inverse of absolute difference)
            diff = abs(avg_retrievals - expected_retrievals)
            max_diff = max(expected_retrievals, 5 - expected_retrievals)
            score = 1.0 - (diff / max_diff)
            
            # Weight by group size
            weighted_score = score * (len(group_results) / len(results))
            scores.append(weighted_score)
        
        # Also consider correlation between difficulty and retrieval count
        if len(results) >= 10:
            difficulties = []
            retrievals = []
            
            for result in results:
                difficulty = result.get("difficulty", "unknown")
                if difficulty != "unknown":
                    # Map difficulty to numeric
                    difficulty_map = {"simple": 1, "medium": 2, "complex": 3, "multi-hop": 4}
                    if difficulty in difficulty_map:
                        difficulties.append(difficulty_map[difficulty])
                        retrievals.append(result.get("retrieval_count", 0))
            
            if len(difficulties) >= 5:
                correlation = np.corrcoef(difficulties, retrievals)[0, 1]
                if not np.isnan(correlation):
                    correlation_score = (correlation + 1) / 2  # Map from [-1, 1] to [0, 1]
                    scores.append(correlation_score * 0.5)  # Weight this component
        
        return np.mean(scores) if scores else 0.5
    
    @staticmethod
    def compute_per_difficulty_metrics(results: List[Dict]) -> Dict[str, Dict[str, float]]:
        """Compute metrics stratified by question difficulty"""
        difficulty_groups = defaultdict(list)
        
        for result in results:
            difficulty = result.get("difficulty", "unknown")
            difficulty_groups[difficulty].append(result)
        
        metrics_by_difficulty = {}
        
        for difficulty, group_results in difficulty_groups.items():
            if difficulty == "unknown":
                continue
            
            if len(group_results) < 1:
                continue
            
            # Compute metrics for this difficulty group
            accuracy = RAGMetrics.compute_accuracy(group_results)
            avg_retrieval_count = RAGMetrics.compute_avg_retrieval_count(group_results)
            avg_latency = RAGMetrics.compute_avg_latency(group_results)
            
            # Compute retrieval efficiency
            expected_retrievals = {
                "simple": 1.0,
                "medium": 2.0,
                "complex": 3.0,
                "multi-hop": 4.0
            }.get(difficulty, 2.0)
            
            retrieval_efficiency = 1.0 - min(1.0, abs(avg_retrieval_count - expected_retrievals) / 3.0)
            
            metrics_by_difficulty[difficulty] = {
                "accuracy": accuracy,
                "avg_retrieval_count": avg_retrieval_count,
                "avg_latency": avg_latency,
                "retrieval_efficiency": retrieval_efficiency,
                "sample_size": len(group_results)
            }
        
        return metrics_by_difficulty
    
    @staticmethod
    def compute_detailed_metrics(results: List[Dict], config: Dict) -> Dict[str, Any]:
        """Compute detailed metrics for analysis"""
        if not results:
            return {}
        
        details = {}
        
        # 1. Action distribution
        actions = []
        for result in results:
            actions.extend(result.get("actions", []))
        
        action_counts = {}
        for action in actions:
            action_counts[action] = action_counts.get(action, 0) + 1
        
        details["action_distribution"] = action_counts
        
        # 2. Confidence analysis
        final_confidences = [r.get("final_confidence", 0.5) for r in results 
                           if "final_confidence" in r]
        if final_confidences:
            details["confidence_stats"] = {
                "mean": np.mean(final_confidences),
                "std": np.std(final_confidences),
                "min": np.min(final_confidences),
                "max": np.max(final_confidences)
            }
        
        # 3. Step count distribution
        step_counts = [r.get("total_steps", 0) for r in results]
        details["step_stats"] = {
            "mean": np.mean(step_counts),
            "std": np.std(step_counts),
            "histogram": np.histogram(step_counts, bins=range(0, 11))[0].tolist()
        }
        
        # 4. Success rate by retrieval count
        retrieval_success = defaultdict(lambda: {"correct": 0, "total": 0})
        for result in results:
            retrievals = result.get("retrieval_count", 0)
            retrieval_success[retrievals]["total"] += 1
            if result.get("correct", False):
                retrieval_success[retrievals]["correct"] += 1
        
        success_rates = {}
        for retrievals, counts in retrieval_success.items():
            if counts["total"] > 0:
                success_rates[retrievals] = counts["correct"] / counts["total"]
        
        details["success_by_retrievals"] = success_rates
        
        # 5. Latency breakdown
        latencies = [r.get("latency", 0.0) for r in results if r.get("latency", 0.0) > 0]
        if latencies:
            details["latency_stats"] = {
                "mean": np.mean(latencies),
                "p50": np.percentile(latencies, 50),
                "p90": np.percentile(latencies, 90),
                "p95": np.percentile(latencies, 95)
            }
        
        # 6. Query reformulation effectiveness
        requery_results = [r for r in results if "re_query" in r.get("actions", [])]
        if requery_results:
            requery_accuracy = RAGMetrics.compute_accuracy(requery_results)
            details["requery_effectiveness"] = {
                "accuracy_with_requery": requery_accuracy,
                "sample_size": len(requery_results)
            }
        
        return details
    
    @staticmethod
    def compute_retrieval_precision(results: List[Dict], top_k: int = 5) -> Dict[str, float]:
        """
        Compute retrieval precision metrics.
        
        This requires ground truth relevant documents in the results.
        """
        precisions_at_k = {}
        
        for k in [1, 3, 5, 10]:
            precisions = []
            
            for result in results:
                if "retrieved_docs" not in result or "relevant_docs" not in result:
                    continue
                
                retrieved = result["retrieved_docs"][:k]
                relevant = set(result["relevant_docs"])
                
                if not relevant:
                    continue
                
                # Compute precision@k
                correct = 0
                for doc in retrieved:
                    # Simple matching
                    if any(relevant_doc in doc for relevant_doc in relevant):
                        correct += 1
                
                precision = correct / min(k, len(retrieved)) if retrieved else 0.0
                precisions.append(precision)
            
            if precisions:
                precisions_at_k[f"precision@{k}"] = np.mean(precisions)
        
        return precisions_at_k
    
    @staticmethod
    def compare_with_baselines(results: List[Dict], baseline_results: Dict[str, List[Dict]]) -> Dict[str, Any]:
        """Compare current results with baseline systems"""
        comparison = {}
        
        # Compute metrics for current system
        current_metrics = RAGMetrics.compute_all_metrics(results, {})
        
        for baseline_name, baseline_runs in baseline_results.items():
            baseline_metrics = RAGMetrics.compute_all_metrics(baseline_runs, {})
            
            comparison[baseline_name] = {
                "accuracy_diff": current_metrics.accuracy - baseline_metrics.accuracy,
                "retrieval_diff": current_metrics.avg_retrieval_count - baseline_metrics.avg_retrieval_count,
                "latency_diff": current_metrics.avg_latency - baseline_metrics.avg_latency,
                "energy_ratio": current_metrics.energy_efficiency / baseline_metrics.energy_efficiency 
                              if baseline_metrics.energy_efficiency > 0 else float("inf")
            }
        
        return comparison
    
    @staticmethod
    def create_visualization_data(results: List[Dict]) -> Dict[str, Any]:
        """Prepare data for visualization"""
        viz_data = {
            "questions": [],
            "retrieval_counts": [],
            "latencies": [],
            "confidences": [],
            "correctness": [],
            "difficulties": []
        }
        
        for result in results:
            viz_data["questions"].append(result.get("question", "")[:50])  # Truncate
            viz_data["retrieval_counts"].append(result.get("retrieval_count", 0))
            viz_data["latencies"].append(result.get("latency", 0.0))
            viz_data["confidences"].append(result.get("final_confidence", 0.5))
            viz_data["correctness"].append(result.get("correct", False))
            viz_data["difficulties"].append(result.get("difficulty", "unknown"))
        
        # Add aggregated data
        viz_data["aggregates"] = {
            "retrieval_by_difficulty": {},
            "accuracy_by_retrieval": {},
            "latency_by_difficulty": {}
        }
        
        # Group by difficulty
        difficulty_groups = defaultdict(list)
        for result in results:
            difficulty = result.get("difficulty", "unknown")
            difficulty_groups[difficulty].append(result)
        
        for difficulty, group in difficulty_groups.items():
            if difficulty != "unknown":
                retrievals = [r.get("retrieval_count", 0) for r in group]
                latencies = [r.get("latency", 0.0) for r in group if r.get("latency", 0.0) > 0]
                
                viz_data["aggregates"]["retrieval_by_difficulty"][difficulty] = {
                    "mean": np.mean(retrievals) if retrievals else 0,
                    "std": np.std(retrievals) if retrievals else 0
                }
                
                if latencies:
                    viz_data["aggregates"]["latency_by_difficulty"][difficulty] = {
                        "mean": np.mean(latencies),
                        "std": np.std(latencies)
                    }
        
        # Accuracy by retrieval count
        retrieval_groups = defaultdict(lambda: {"correct": 0, "total": 0})
        for result in results:
            retrievals = result.get("retrieval_count", 0)
            retrieval_groups[retrievals]["total"] += 1
            if result.get("correct", False):
                retrieval_groups[retrievals]["correct"] += 1
        
        for retrievals, counts in retrieval_groups.items():
            if counts["total"] > 0:
                viz_data["aggregates"]["accuracy_by_retrieval"][retrievals] = {
                    "accuracy": counts["correct"] / counts["total"],
                    "samples": counts["total"]
                }
        
        return viz_data


class MetricLogger:
    """Logger for tracking metrics during training/evaluation"""
    
    def __init__(self, log_dir: str = None):
        self.log_dir = log_dir
        self.metrics_history = defaultdict(list)
        self.current_epoch = 0
        
    def log_metric(self, name: str, value: float, step: int = None):
        """Log a metric value"""
        if step is None:
            step = self.current_epoch
        
        self.metrics_history[name].append({
            "step": step,
            "value": value,
            "timestamp": time.time()
        })
        
        # Print to console
        print(f"[Metric] {name}: {value:.4f} (step {step})")
    
    def log_metrics_batch(self, metrics: Dict[str, float], step: int = None):
        """Log multiple metrics at once"""
        for name, value in metrics.items():
            self.log_metric(name, value, step)
    
    def get_metric_history(self, name: str) -> List[Dict]:
        """Get history for a specific metric"""
        return self.metrics_history.get(name, [])
    
    def get_best_metric(self, name: str, higher_is_better: bool = True) -> Dict:
        """Get the best value for a metric"""
        history = self.get_metric_history(name)
        if not history:
            return None
        
        if higher_is_better:
            best_entry = max(history, key=lambda x: x["value"])
        else:
            best_entry = min(history, key=lambda x: x["value"])
        
        return best_entry
    
    def save_logs(self, filepath: str):
        """Save metrics to file"""
        with open(filepath, "w") as f:
            json.dump(self.metrics_history, f, indent=2, default=str)
    
    def load_logs(self, filepath: str):
        """Load metrics from file"""
        with open(filepath, "r") as f:
            self.metrics_history = json.load(f)
    
    def create_summary(self) -> Dict[str, Any]:
        """Create a summary of all metrics"""
        summary = {
            "metrics": {},
            "best_values": {}
        }
        
        for metric_name, history in self.metrics_history.items():
            if history:
                values = [entry["value"] for entry in history]
                summary["metrics"][metric_name] = {
                    "mean": np.mean(values),
                    "std": np.std(values),
                    "min": np.min(values),
                    "max": np.max(values),
                    "latest": values[-1],
                    "history_length": len(values)
                }
                
                higher_is_better = any(word in metric_name.lower() 
                                      for word in ["accuracy", "reward", "score", "f1"])
                
                if higher_is_better:
                    best = max(history, key=lambda x: x["value"])
                else:
                    best = min(history, key=lambda x: x["value"])
                
                summary["best_values"][metric_name] = best
        
        return summary