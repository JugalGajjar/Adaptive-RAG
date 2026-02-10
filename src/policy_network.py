import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

class AdaptivePolicyNetwork(nn.Module):
    """Policy network for adaptive retrieval decisions"""
    
    def __init__(self, config, state_dim: int):
        super().__init__()
        self.config = config
        
        # Feature extraction layers
        self.state_encoder = nn.Sequential(
            nn.Linear(state_dim, config.model.policy_network.hidden_size),
            nn.ReLU(),
            nn.Dropout(config.model.policy_network.dropout),
            nn.Linear(config.model.policy_network.hidden_size, 
                     config.model.policy_network.hidden_size),
            nn.ReLU()
        )
        
        # Attention layer for focusing on important state features
        if config.model.policy_network.use_attention:
            self.attention = nn.MultiheadAttention(
                embed_dim=config.model.policy_network.hidden_size,
                num_heads=4,
                dropout=config.model.policy_network.dropout,
                batch_first=True
            )
        
        # Decision head
        self.action_head = nn.Sequential(
            nn.Linear(config.model.policy_network.hidden_size, 64),
            nn.ReLU(),
            nn.Linear(64, 3)  # 3 actions
        )
        
        # Value head for baseline (if using actor-critic)
        self.value_head = nn.Sequential(
            nn.Linear(config.model.policy_network.hidden_size, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        
        # Confidence predictor (auxiliary task)
        self.confidence_predictor = nn.Sequential(
            nn.Linear(config.model.policy_network.hidden_size, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
    
    def forward(self, state: torch.Tensor, 
                return_attention: bool = False) -> Tuple[torch.Tensor, ...]:
        """
        Forward pass through policy network
        
        Args:
            state: [batch_size, state_dim]
            return_attention: whether to return attention weights
            
        Returns:
            action_logits: [batch_size, 3]
            state_value: [batch_size, 1]
            confidence_pred: [batch_size, 1]
            attention_weights: optional
        """
        batch_size = state.size(0)
        
        # Encode state
        encoded = self.state_encoder(state)  # [batch_size, hidden_size]
        
        # Apply attention if enabled
        attention_weights = None
        if self.config.model.policy_network.use_attention:
            # Reshape for attention
            encoded = encoded.unsqueeze(1)  # [batch_size, 1, hidden_size]
            encoded, attention_weights = self.attention(
                encoded, encoded, encoded
            )
            encoded = encoded.squeeze(1)
        
        # Get action probabilities
        action_logits = self.action_head(encoded)
        
        # Get state value (for baseline)
        state_value = self.value_head(encoded)
        
        # Predict confidence (auxiliary task)
        confidence_pred = self.confidence_predictor(encoded)
        
        if return_attention:
            return action_logits, state_value, confidence_pred, attention_weights
        return action_logits, state_value, confidence_pred
    
    def get_action(self, state: torch.Tensor, 
                   epsilon: float = 0.1,
                   training: bool = True) -> Tuple[int, torch.Tensor]:
        """
        Select action using epsilon-greedy policy
        
        Args:
            state: current state tensor
            epsilon: exploration rate
            training: whether in training mode
            
        Returns:
            action: selected action index
            log_prob: log probability of selected action
        """
        with torch.no_grad() if not training else torch.enable_grad():
            action_logits, _, _ = self.forward(state)
            action_probs = F.softmax(action_logits, dim=-1)
            
            if training and torch.rand(1) < epsilon:
                # Explore: random action
                action = torch.randint(0, 3, (1,)).item()
                log_prob = torch.log(action_probs[0, action] + 1e-10)
            else:
                # Exploit: best action
                action = torch.argmax(action_probs, dim=-1).item()
                log_prob = torch.log(action_probs[0, action] + 1e-10)
            
            return action, log_prob
    
    def get_action_distribution(self, state: torch.Tensor) -> torch.Tensor:
        """Get full action probability distribution"""
        action_logits, _, _ = self.forward(state)
        return F.softmax(action_logits, dim=-1)