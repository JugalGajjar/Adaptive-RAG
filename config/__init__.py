"""
Configuration loading utilities.
"""

import yaml
from omegaconf import DictConfig, OmegaConf
from pathlib import Path
from typing import Dict, Any


def load_config(config_path: str = "config/config.yaml") -> DictConfig:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        DictConfig: Configuration object
    """
    config_path = Path(config_path)
    
    if not config_path.exists():
        # Create default config if doesn't exist
        default_config = {
            "model": {
                "generator": {"type": "huggingface", "name": "Qwen/Qwen3-4B-Instruct-2507"},
                "retriever": {"type": "dense", "embedding_model": "all-MiniLM-L6-v2"},
                "policy_network": {"hidden_size": 256}
            },
            "environment": {"max_retrieval_steps": 5},
            "data": {"train_path": "data/train.jsonl"}
        }
        
        config_path.parent.mkdir(parents=True, exist_ok=True)
        with open(config_path, "w") as f:
            yaml.dump(default_config, f)
        
        print(f"Created default config at {config_path}")
    
    # Load configuration
    config = OmegaConf.load(config_path)
    
    # Set defaults for missing values
    defaults = {
        "model": {
            "generator": {"type": "huggingface", "max_tokens": 512, "temperature": 0.1},
            "retriever": {"type": "dense", "top_k_initial": 3},
            "policy_network": {"hidden_size": 256, "num_layers": 2, "dropout": 0.1}
        },
        "training": {
            "batch_size": 32,
            "learning_rate": 1e-4,
            "reward_weights": {
                "correctness": 5.0,
                "retrieval_penalty": -0.1,
                "latency_penalty": -0.01,
                "step_penalty": -0.05,
                "retrieval_quality": 2.0
            }
        },
        "environment": {
            "max_retrieval_steps": 5,
            "confidence_threshold": 0.7
        },
        "evaluation": {
            "checkpoint_path": "checkpoints/best_model.pt"
        }
    }
    
    # Merge with defaults
    config = OmegaConf.merge(defaults, config)
    
    return config


def save_config(config: DictConfig, config_path: str):
    """Save configuration to file"""
    with open(config_path, "w") as f:
        OmegaConf.save(config, f)


def get_config_value(config: DictConfig, key_path: str, default: Any = None) -> Any:
    """Get configuration value with dot notation"""
    try:
        return OmegaConf.select(config, key_path)
    except:
        return default