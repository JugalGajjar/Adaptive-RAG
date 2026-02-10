"""
Evaluate Fixed RAG baseline
"""

import hydra
from omegaconf import DictConfig
from pathlib import Path
import json
import time

from src.baselines.fixed_rag import FixedRAG
from src.data_processor import DatasetLoader
from src.utils.model_initializers import initialize_generator, initialize_retriever


@hydra.main(version_base=None, config_path="../config", config_name="config_fixed_rag")
def main(cfg: DictConfig):
    print("Evaluating Fixed RAG baseline...")
    
    # Load data
    data_loader = DatasetLoader(cfg.data)
    test_data = data_loader.load_test()
    
    # Initialize components
    generator = initialize_generator(cfg.model.generator)
    retriever = initialize_retriever(cfg.model.retriever, data_loader.corpus)
    
    # Create Fixed RAG
    fixed_count = cfg.fixed_rag.fixed_retrieval_count
    fixed_rag = FixedRAG(cfg, generator, retriever, fixed_count=fixed_count)
    
    # Evaluate
    results = []
    for i, item in enumerate(test_data[:cfg.evaluation.max_samples]):
        answer, retrievals, latency = fixed_rag.answer(item.question)
        
        # Check correctness
        correct = check_correctness(answer, item.answer)
        
        results.append({
            "question_id": item.id,
            "question": item.question,
            "answer": answer,
            "reference": item.answer,
            "correct": correct,
            "retrieval_count": retrievals,
            "latency": latency,
            "difficulty": item.difficulty
        })
        
        if (i + 1) % 10 == 0:
            print(f"Evaluated {i + 1}/{len(test_data)} questions...")
    
    # Compute metrics
    accuracy = sum(r["correct"] for r in results) / len(results)
    avg_retrievals = sum(r["retrieval_count"] for r in results) / len(results)
    avg_latency = sum(r["latency"] for r in results) / len(results)
    
    print(f"\nResults:")
    print(f"  Accuracy: {accuracy:.3f}")
    print(f"  Avg Retrievals: {avg_retrievals:.2f}")
    print(f"  Avg Latency: {avg_latency:.3f}s")
    
    # Save results
    output_dir = Path(cfg.evaluation.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(output_dir / "results.json", "w") as f:
        json.dump({
            "accuracy": accuracy,
            "avg_retrievals": avg_retrievals,
            "avg_latency": avg_latency,
            "detailed_results": results
        }, f, indent=2)
    
    print(f"Results saved to {output_dir}")


def check_correctness(answer: str, reference: str) -> bool:
    """Check if answer is correct"""
    if not answer or not reference:
        return False
    answer_lower = answer.lower().strip()
    reference_lower = reference.lower().strip()
    return answer_lower == reference_lower or reference_lower in answer_lower


if __name__ == "__main__":
    main()