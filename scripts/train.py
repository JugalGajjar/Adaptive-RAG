"""
Training script for Adaptive Retrieval RAG
FIXED: State dimension mismatch bug
"""

import hydra
from omegaconf import DictConfig, OmegaConf
import torch
import wandb
import numpy as np
import random
from pathlib import Path
import logging
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
    # Setup
    print(f"Config:\n{OmegaConf.to_yaml(cfg)}")
    
    # Set random seeds
    torch.manual_seed(cfg.training.seed)
    np.random.seed(cfg.training.seed)
    random.seed(cfg.training.seed)
    
    # Initialize WandB
    if cfg.wandb.enabled:
        wandb.init(
            project=cfg.wandb.project,
            name=cfg.wandb.name,
            config=OmegaConf.to_container(cfg, resolve=True)
        )
    
    # Load data
    print("Loading data...")
    data_loader = DatasetLoader(cfg.data)
    train_data = data_loader.load_train()
    val_data = data_loader.load_val()
    
    print(f"Loaded {len(train_data)} training examples")
    print(f"Loaded {len(val_data)} validation examples")
    
    # Initialize components
    print("Initializing components...")
    
    # Initialize generator and retriever
    generator = initialize_generator(cfg.model.generator)
    retriever = initialize_retriever(cfg.model.retriever, data_loader.corpus)
    
    # Initialize reward calculator
    reward_calculator = AdaptiveRAGRewardCalculator(cfg)
    
    # Initialize environment WITH data_loader
    env = RAGEnvironment(
        cfg, 
        generator, 
        retriever, 
        reward_calculator,
        data_loader=data_loader
    )
    
    # Set environment to training mode
    env.set_mode('train')
    
    # CRITICAL FIX: Get actual state dimension from environment
    print("\nDetecting actual state dimensions...")
    test_state = env.reset()
    actual_state_dim = len(test_state) if hasattr(test_state, '__len__') else test_state.shape[0]
    reported_state_dim = env.observation_space.shape[0]
    
    print(f"   Reported by observation_space: {reported_state_dim}")
    print(f"   Actual from env.reset(): {actual_state_dim}")
    
    if actual_state_dim != reported_state_dim:
        print(f"MISMATCH DETECTED!")
        print(f"   Using ACTUAL dimension: {actual_state_dim}")
        state_dim = actual_state_dim
    else:
        print(f"Dimensions match: {actual_state_dim}")
        state_dim = actual_state_dim
    
    # Initialize policy network with CORRECT dimension
    print(f"\nCreating policy network with state_dim={state_dim}...")
    policy_net = AdaptivePolicyNetwork(cfg, state_dim)
    
    # Initialize agent
    device = get_best_device()
    agent = AdaptiveRAGAgent(cfg, policy_net, env, device)
    
    # Verify by testing forward pass
    print("Testing policy network forward pass...")
    try:
        test_state_tensor = torch.FloatTensor(test_state).unsqueeze(0).to(device)
        test_output = agent.policy_net(test_state_tensor)
        print(f"Forward pass successful! Output shapes: {[o.shape for o in test_output]}")
    except Exception as e:
        print(f"Forward pass failed: {e}")
        print("   This indicates the dimension fix didn't work. Aborting.")
        raise
    
    # Warm-start with supervised data
    if cfg.training.warmup_steps > 0:
        print("\n" + "="*60)
        print("WARM-START PHASE")
        print("="*60)
        warmup_data = data_loader.load_warmup()
        agent.warm_start(warmup_data, cfg.training.warmup_steps)
    
    # Main training loop
    print("\n" + "="*60)
    print("MAIN TRAINING PHASE")
    print("="*60)
    print(f"Total episodes: {cfg.training.total_episodes}")
    print(f"Episodes per epoch: {cfg.training.episodes_per_epoch}")
    print(f"Evaluation interval: {cfg.training.eval_interval}")
    print("="*60 + "\n")
    
    best_val_reward = -float('inf')
    
    pbar = tqdm(range(cfg.training.total_episodes), desc="Training")
    
    for episode in pbar:
        # Training phase - environment already in 'train' mode
        agent.collect_experience(num_episodes=cfg.training.episodes_per_epoch)
        
        # Train on collected experience
        for _ in range(cfg.training.updates_per_epoch):
            agent.train_step(batch_size=cfg.training.batch_size)
        
        # Update progress bar
        avg_reward = np.mean(agent.episode_rewards[-10:]) if agent.episode_rewards else 0
        avg_retrievals = np.mean(agent.retrieval_counts[-10:]) if agent.retrieval_counts else 0
        
        pbar.set_postfix({
            'reward': f'{avg_reward:.2f}',
            'ret': f'{avg_retrievals:.1f}',
            'eps': f'{agent.epsilon:.3f}'
        })
        
        # Evaluate periodically
        if episode % cfg.training.eval_interval == 0 and episode > 0:
            # Switch to validation mode
            env.set_mode('val')
            
            val_reward = evaluate(agent, val_data, cfg)
            
            # Switch back to training mode
            env.set_mode('train')
            
            if val_reward > best_val_reward:
                best_val_reward = val_reward
                # Save best model
                checkpoint_dir = Path(cfg.training.checkpoint_dir)
                import os
                os.makedirs(cfg.training.checkpoint_dir, exist_ok=True)
                checkpoint_dir.mkdir(parents=True, exist_ok=True)
                checkpoint_path = checkpoint_dir / "best_model.pt"
                agent.save_checkpoint(checkpoint_path)
                print(f"\nNew best model saved with reward: {val_reward:.3f}")
            
            # Log evaluation metrics
            if cfg.wandb.enabled:
                wandb.log({
                    'val/reward': val_reward,
                    'val/best_reward': best_val_reward,
                    'episode': episode
                })
        
        # Save periodic checkpoint
        if episode % cfg.training.checkpoint_interval == 0 and episode > 0:
            import os
            os.makedirs(cfg.training.checkpoint_dir, exist_ok=True)
            checkpoint_dir = Path(cfg.training.checkpoint_dir)
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            checkpoint_path = checkpoint_dir / f"checkpoint_ep{episode}.pt"
            agent.save_checkpoint(checkpoint_path)
            print(f"\nCheckpoint saved at episode {episode}")
    
    print("\n" + "="*60)
    print("TRAINING COMPLETED!")
    print("="*60)
    
    # Final evaluation
    env.set_mode('val')
    final_reward = evaluate(agent, val_data, cfg)
    print(f"Final validation reward: {final_reward:.3f}")
    
    # Save final model
    final_path = Path(cfg.training.checkpoint_dir) / "final_model.pt"
    agent.save_checkpoint(final_path)
    print(f"Final model saved to: {final_path}")
    
    if cfg.wandb.enabled:
        wandb.finish()


def evaluate(agent, val_data, cfg):
    """
    Evaluate agent on validation data.
    
    NOTE: Environment should already be in 'val' mode before calling this.
    """
    total_reward = 0
    total_retrievals = 0
    total_correct = 0
    
    num_samples = min(len(val_data), cfg.evaluation.num_val_samples)
    
    print(f"\n📊 Evaluating on {num_samples} validation samples...")
    
    for i in tqdm(range(num_samples), desc="Validation", leave=False):
        # Environment will sample from val split
        state = agent.env.reset()
        
        # Get reference answer from environment's stored metadata
        reference = agent.env.get_current_reference_answer()
        
        episode_reward = 0
        episode_retrievals = 0
        
        while True:
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(agent.device)
            action, _ = agent.policy_net.get_action(
                state_tensor, 
                epsilon=0.0,  # No exploration during evaluation
                training=False
            )
            
            next_state, reward, done, info = agent.env.step(action)
            
            episode_reward += reward
            if info.get('action') in ['retrieve_more', 're_query']:
                episode_retrievals += 1
            
            state = next_state
            
            if done:
                # Check correctness
                answer = info.get('final_answer', '')
                if reference and check_answer_correct(answer, reference):
                    total_correct += 1
                break
        
        total_reward += episode_reward
        total_retrievals += episode_retrievals
    
    # Calculate averages
    avg_reward = total_reward / num_samples
    avg_retrievals = total_retrievals / num_samples
    accuracy = total_correct / num_samples
    
    print(f"\n📈 Validation Results:")
    print(f"   Avg Reward: {avg_reward:.3f}")
    print(f"   Avg Retrievals: {avg_retrievals:.2f}")
    print(f"   Accuracy: {accuracy:.3f}")
    
    # Log detailed metrics if wandb is enabled
    if wandb.run is not None:
        wandb.log({
            'val/avg_reward': avg_reward,
            'val/avg_retrievals': avg_retrievals,
            'val/accuracy': accuracy
        })
    
    return avg_reward


def check_answer_correct(answer: str, reference: str) -> bool:
    """Check if answer is correct (simple string matching)"""
    if not answer or not reference:
        return False
    
    answer_lower = answer.lower().strip()
    reference_lower = reference.lower().strip()
    
    # Exact match or reference contained in answer
    return answer_lower == reference_lower or reference_lower in answer_lower


if __name__ == "__main__":
    main()