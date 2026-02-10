"""
Adaptive RAG Agent
"""

import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from collections import deque
import random
from typing import List, Tuple, Optional
import wandb
import logging
from tqdm import tqdm

from src.policy_network import AdaptivePolicyNetwork
from src.data_processor import DataPoint

logger = logging.getLogger(__name__)


class AdaptiveRAGAgent:
    """Agent that learns adaptive retrieval policy"""
    
    def __init__(self, config, policy_net, env, device="cuda"):
        self.config = config
        self.env = env
        self.device = device
        
        # Move policy network to device
        self.policy_net = policy_net.to(device)
        
        # Store state_dim from policy_net for target network
        self.state_dim = self._get_policy_net_state_dim()
        
        # Create target network
        self.target_net = self._create_target_network().to(device)
        
        # Optimization
        self.optimizer = optim.Adam(
            self.policy_net.parameters(),
            lr=config.training.learning_rate
        )
        
        # Experience replay
        self.replay_buffer = deque(maxlen=config.training.buffer_size)
        
        # Tracking
        self.episode_rewards = []
        self.retrieval_counts = []
        
        # Training state
        self.step_count = 0
        self.epsilon = 1.0  # Initial exploration rate
        self.epsilon_decay = 0.9995
        self.epsilon_min = 0.01
    
    def _get_policy_net_state_dim(self):
        """
        Extract state_dim from existing policy_net.
        The policy_net was created with correct dimension in train.py
        """
        # Get the input dimension from first layer of state_encoder
        first_layer = self.policy_net.state_encoder[0]  # First Linear layer
        state_dim = first_layer.in_features
        
        logger.info(f"Extracted state_dim from policy_net: {state_dim}")
        return state_dim
        
    def _create_target_network(self):
        """
        Create target network for stable learning.
        FIXED: Use self.state_dim instead of env.observation_space
        """
        target_net = AdaptivePolicyNetwork(
            self.config, 
            self.state_dim  # Use extracted dimension
        )
        target_net.load_state_dict(self.policy_net.state_dict())
        return target_net
    
    def collect_experience(self, num_episodes: int = 10):
        """
        Collect experience from environment.
        Environment handles question sampling based on its mode (train/val/test).
        """
        for episode in tqdm(range(num_episodes), desc="Collecting experience", leave=False):
            # Reset environment - it will sample question based on current mode
            state = self.env.reset()
            
            # Get metadata from environment's sampled question
            reference_answer = self.env.get_current_reference_answer()
            question_id = self.env.get_current_question_id()
            difficulty = self.env.get_current_difficulty()
            has_context = len(self.env.relevant_docs) > 0
            
            episode_reward = 0
            episode_retrievals = 0
            episode_steps = 0
            
            while True:
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                
                # Select action
                action, log_prob = self.policy_net.get_action(
                    state_tensor, 
                    epsilon=self.epsilon,
                    training=True
                )
                
                # Take action in environment
                next_state, reward, done, info = self.env.step(action)
                
                # Store experience with metadata
                experience = (
                    state,
                    action,
                    reward,
                    next_state,
                    done,
                    {
                        **info,
                        "reference_answer": reference_answer,
                        "question_id": question_id,
                        "difficulty": difficulty,
                        "has_context": has_context,
                        "confidence": self.env.current_confidence
                    }
                )
                self.replay_buffer.append(experience)
                
                # Update counts
                episode_reward += reward
                episode_steps += 1
                if info.get("action") in ["retrieve_more", "re_query"]:
                    episode_retrievals += 1
                
                # Update state
                state = next_state
                
                if done:
                    break
            
            # Decay epsilon
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
            
            # Record episode stats
            self.episode_rewards.append(episode_reward)
            self.retrieval_counts.append(episode_retrievals)
            
            # Logging (reduced frequency to avoid clutter)
            if episode % 10 == 0:
                logger.debug(
                    f"Episode {episode}: Reward={episode_reward:.2f}, "
                    f"Retrievals={episode_retrievals}, Steps={episode_steps}, "
                    f"Epsilon={self.epsilon:.3f}"
                )
                
                if wandb.run is not None:
                    wandb.log({
                        "train/episode_reward": episode_reward,
                        "train/episode_retrievals": episode_retrievals,
                        "train/episode_steps": episode_steps,
                        "train/epsilon": self.epsilon,
                        "step": self.step_count
                    })
    
    def train_step(self, batch_size: int = 32):
        """Perform one training step using sampled batch"""
        if len(self.replay_buffer) < batch_size:
            return
        
        # Sample batch
        batch = random.sample(self.replay_buffer, batch_size)
        states, actions, rewards, next_states, dones, infos = zip(*batch)
        
        # Convert to tensors
        states_t = torch.FloatTensor(np.array(states)).to(self.device)
        actions_t = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards_t = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states_t = torch.FloatTensor(np.array(next_states)).to(self.device)
        dones_t = torch.FloatTensor(dones).unsqueeze(1).to(self.device)
        
        # Get current Q-values
        current_q, current_v, current_conf = self.policy_net(states_t)
        current_q = current_q.gather(1, actions_t)
        
        # Get target Q-values
        with torch.no_grad():
            next_q, next_v, _ = self.target_net(next_states_t)
            next_q_max = next_q.max(1)[0].unsqueeze(1)
            target_q = rewards_t + (1 - dones_t) * self.config.training.gamma * next_q_max
        
        # Compute losses
        q_loss = F.mse_loss(current_q, target_q)
        v_loss = F.mse_loss(current_v, target_q.detach())
        
        # Auxiliary loss: confidence prediction
        actual_confidences = torch.FloatTensor([
            info.get("confidence", 0.5) for info in infos
        ]).unsqueeze(1).to(self.device)
        conf_loss = F.mse_loss(current_conf, actual_confidences)
        
        # Total loss
        total_loss = q_loss + 0.5 * v_loss + 0.1 * conf_loss
        
        # Optimize
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()
        
        # Update target network
        self._soft_update_target_network()
        
        # Increment step count
        self.step_count += 1
        
        # Log training metrics
        if self.step_count % 100 == 0 and wandb.run is not None:
            wandb.log({
                "train/q_loss": q_loss.item(),
                "train/v_loss": v_loss.item(),
                "train/conf_loss": conf_loss.item(),
                "train/total_loss": total_loss.item(),
                "step": self.step_count
            })
    
    def _soft_update_target_network(self):
        """Soft update target network parameters"""
        tau = self.config.training.tau
        for target_param, param in zip(self.target_net.parameters(), 
                                       self.policy_net.parameters()):
            target_param.data.copy_(
                tau * param.data + (1 - tau) * target_param.data
            )
    
    def warm_start(self, warmup_data: List[DataPoint], num_steps: int):
        """
        Confidence-based warm-start.
        Policy learns to retrieve until confident, not until fixed count reached.
        """
        logger.info(f"Starting warm-start: {len(warmup_data)} examples, {num_steps} steps")
        
        for step in tqdm(range(num_steps), desc="Warmup", unit="steps"):
            data_point = random.choice(warmup_data)
            
            # Get target outcomes
            target_confidence = data_point.metadata.get("target_confidence", 0.85)
            min_retrievals = data_point.metadata.get("min_retrievals", 1)
            max_retrievals = data_point.metadata.get("max_retrievals", 6)
            
            # Reset environment
            state = self.env.reset(
                question=data_point.question,
                relevant_docs=data_point.context
            )
            
            episode_reward = 0
            retrieval_count = 0
            
            while True:
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                
                # Let policy decide (with exploration)
                action, log_prob = self.policy_net.get_action(
                    state_tensor,
                    epsilon=0.3,  # Explore during warmup
                    training=True
                )
                
                # Safety: enforce minimum retrievals
                if retrieval_count < min_retrievals and action == 2:  # answer_now
                    action = 0  # retrieve_more
                
                # Soft guidance for very low confidence
                if self.env.current_confidence < 0.5 and action == 2 and retrieval_count < 2:
                    if random.random() < 0.5:
                        action = 0  # suggest retrieve_more
                
                # Take action
                next_state, reward, done, info = self.env.step(action)
                
                # Store experience
                experience = (
                    state, action, reward, next_state, done,
                    {
                        **info,
                        "warmup": True,
                        "target_confidence": target_confidence,
                        "current_confidence": self.env.current_confidence
                    }
                )
                self.replay_buffer.append(experience)
                
                episode_reward += reward
                if info.get("action") in ["retrieve_more", "re_query"]:
                    retrieval_count += 1
                
                state = next_state
                
                # Stop conditions
                if done or retrieval_count >= max_retrievals:
                    break
            
            # Train on experience
            if len(self.replay_buffer) >= self.config.training.batch_size:
                self.train_step(batch_size=self.config.training.batch_size)
            
            # Log progress (reduced frequency)
            if step % 100 == 0 and step > 0:
                logger.info(
                    f"Warmup {step}/{num_steps}: "
                    f"Reward={episode_reward:.2f}, Retrievals={retrieval_count}, "
                    f"Conf={self.env.current_confidence:.2f}"
                )
        
        logger.info("Warm-start completed!")
    
    def save_checkpoint(self, path: str):
        """Save agent checkpoint"""
        checkpoint = {
            "policy_state_dict": self.policy_net.state_dict(),
            "target_state_dict": self.target_net.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "step_count": self.step_count,
            "epsilon": self.epsilon,
            "state_dim": self.state_dim,  # Save for verification
            "episode_rewards": self.episode_rewards[-100:],
            "retrieval_counts": self.retrieval_counts[-100:],
            "config": self.config
        }
        torch.save(checkpoint, path)
        logger.info(f"Checkpoint saved to {path}")
    
    def load_checkpoint(self, path: str):
        """Load agent checkpoint"""
        checkpoint = torch.load(path, map_location=self.device)
        
        # Verify state dimension matches
        if "state_dim" in checkpoint and checkpoint["state_dim"] != self.state_dim:
            logger.warning(
                f"State dimension mismatch in checkpoint! "
                f"Checkpoint: {checkpoint['state_dim']}, Current: {self.state_dim}"
            )
        
        self.policy_net.load_state_dict(checkpoint["policy_state_dict"])
        self.target_net.load_state_dict(checkpoint["target_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.step_count = checkpoint["step_count"]
        self.epsilon = checkpoint["epsilon"]
        self.episode_rewards = checkpoint.get("episode_rewards", [])
        self.retrieval_counts = checkpoint.get("retrieval_counts", [])
        logger.info(f"Checkpoint loaded from {path} (step {self.step_count})")