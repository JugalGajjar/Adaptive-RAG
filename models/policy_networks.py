"""
Alternative policy network architectures for Adaptive Retrieval RAG.
Provides different neural network architectures for the policy.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List


class SimpleMLPPolicy(nn.Module):
    """Simple Multi-Layer Perceptron policy network"""
    
    def __init__(self, state_dim: int, hidden_dims: List[int] = [256, 128], dropout: float = 0.1):
        super().__init__()
        
        layers = []
        input_dim = state_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            input_dim = hidden_dim
        
        self.feature_extractor = nn.Sequential(*layers)
        
        # Output heads
        self.action_head = nn.Linear(input_dim, 3)  # 3 actions
        self.value_head = nn.Linear(input_dim, 1)   # State value
        self.confidence_head = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
    
    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        features = self.feature_extractor(state)
        
        action_logits = self.action_head(features)
        state_value = self.value_head(features)
        confidence = self.confidence_head(features)
        
        return action_logits, state_value, confidence


class TransformerPolicy(nn.Module):
    """Transformer-based policy network for sequential state reasoning"""
    
    def __init__(self, state_dim: int, hidden_dim: int = 256, num_heads: int = 4, 
                 num_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim
        
        # Project state to transformer dimension
        self.state_projection = nn.Linear(state_dim, hidden_dim)
        
        # Positional encoding for state components
        self.positional_encoding = nn.Parameter(torch.zeros(1, 10, hidden_dim))
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Pooling layer
        self.pooling = nn.AdaptiveAvgPool1d(1)
        
        # Output heads
        self.action_head = nn.Linear(hidden_dim, 3)
        self.value_head = nn.Linear(hidden_dim, 1)
        self.confidence_head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
    
    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size = state.size(0)
        
        # Reshape state into sequence of features
        seq_len = 10
        feature_dim = self.state_dim // seq_len
        
        if feature_dim * seq_len != self.state_dim:
            # Pad if needed
            padding = seq_len * feature_dim - self.state_dim
            if padding > 0:
                state = F.pad(state, (0, padding))
            elif padding < 0:
                state = state[:, :seq_len * feature_dim]
        
        # Reshape to sequence
        state_seq = state.view(batch_size, seq_len, feature_dim)
        
        # Project to transformer dimension
        projected = self.state_projection(state_seq)
        
        # Add positional encoding
        projected = projected + self.positional_encoding[:, :seq_len, :]
        
        # Apply transformer
        transformer_out = self.transformer(projected)
        
        # Pool across sequence dimension
        pooled = self.pooling(transformer_out.transpose(1, 2)).squeeze(-1)
        
        # Output heads
        action_logits = self.action_head(pooled)
        state_value = self.value_head(pooled)
        confidence = self.confidence_head(pooled)
        
        return action_logits, state_value, confidence


class AttentionPolicy(nn.Module):
    """Policy network with multi-head attention over state components"""
    
    def __init__(self, state_dim: int, hidden_dim: int = 256, num_heads: int = 4, 
                 dropout: float = 0.1):
        super().__init__()
        
        # State component projections (assume 6 components)
        self.component_dims = {
            "question": 384,
            "partial_answer": 384,
            "retrieved_docs": 128,
            "scalar": 8,
            "history": 16,
        }
        
        # Project each component
        self.projections = nn.ModuleDict()
        for name, dim in self.component_dims.items():
            self.projections[name] = nn.Sequential(
                nn.Linear(dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            )
        
        # Multi-head attention
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Layer normalization
        self.norm = nn.LayerNorm(hidden_dim)
        
        # Output heads
        self.action_head = nn.Linear(hidden_dim, 3)
        self.value_head = nn.Linear(hidden_dim, 1)
        self.confidence_head = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
    
    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size = state.size(0)
        
        # Split state into components
        components = {}
        start_idx = 0
        for name, dim in self.component_dims.items():
            if start_idx + dim <= state.size(1):
                components[name] = state[:, start_idx:start_idx + dim]
                start_idx += dim
            else:
                # Pad if needed
                pad_size = dim - (state.size(1) - start_idx)
                component = F.pad(state[:, start_idx:], (0, pad_size))
                components[name] = component
                start_idx = state.size(1)
        
        # Project each component
        projected_components = []
        for name in self.component_dims.keys():
            if name in components:
                projected = self.projections[name](components[name])
                projected_components.append(projected.unsqueeze(1))
        
        # Concatenate along sequence dimension
        if projected_components:
            sequence = torch.cat(projected_components, dim=1)
        else:
            # Fallback: use state directly
            sequence = state.unsqueeze(1)
            if sequence.size(2) != self.component_dims["question"]:
                sequence = F.pad(sequence, (0, self.component_dims["question"] - sequence.size(2)))
        
        # Apply attention
        attended, attention_weights = self.attention(sequence, sequence, sequence)
        attended = self.norm(attended)
        
        # Mean pooling across components
        pooled = attended.mean(dim=1)
        
        # Output heads
        action_logits = self.action_head(pooled)
        state_value = self.value_head(pooled)
        confidence = self.confidence_head(pooled)
        
        return action_logits, state_value, confidence


class ResidualPolicy(nn.Module):
    """Policy network with residual connections for stable learning"""
    
    def __init__(self, state_dim: int, hidden_dim: int = 256, num_blocks: int = 3, 
                 dropout: float = 0.1):
        super().__init__()
        
        # Initial projection
        self.input_projection = nn.Linear(state_dim, hidden_dim)
        
        # Residual blocks
        self.blocks = nn.ModuleList()
        for _ in range(num_blocks):
            block = ResidualBlock(hidden_dim, dropout)
            self.blocks.append(block)
        
        # Layer normalization
        self.norm = nn.LayerNorm(hidden_dim)
        
        # Output heads
        self.action_head = nn.Linear(hidden_dim, 3)
        self.value_head = nn.Linear(hidden_dim, 1)
        self.confidence_head = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
    
    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Initial projection
        x = self.input_projection(state)
        
        # Apply residual blocks
        for block in self.blocks:
            x = block(x)
        
        # Normalize
        x = self.norm(x)
        
        # Output heads
        action_logits = self.action_head(x)
        state_value = self.value_head(x)
        confidence = self.confidence_head(x)
        
        return action_logits, state_value, confidence


class ResidualBlock(nn.Module):
    """Residual block with layer normalization"""
    
    def __init__(self, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        
        self.layers = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Dropout(dropout)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.layers(x)


class EnsemblePolicy(nn.Module):
    """Ensemble of policy networks for more robust decisions"""
    
    def __init__(self, state_dim: int, num_models: int = 3, model_type: str = "mlp"):
        super().__init__()
        
        self.num_models = num_models
        
        # Create ensemble of models
        self.models = nn.ModuleList()
        for _ in range(num_models):
            if model_type == "mlp":
                model = SimpleMLPPolicy(state_dim)
            elif model_type == "transformer":
                model = TransformerPolicy(state_dim)
            elif model_type == "attention":
                model = AttentionPolicy(state_dim)
            elif model_type == "residual":
                model = ResidualPolicy(state_dim)
            else:
                raise ValueError(f"Unknown model type: {model_type}")
            self.models.append(model)
        
        # Ensemble combination weights (learnable)
        self.ensemble_weights = nn.Parameter(torch.ones(num_models) / num_models)
    
    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Get outputs from all models
        all_action_logits = []
        all_values = []
        all_confidences = []
        
        for model in self.models:
            action_logits, value, confidence = model(state)
            all_action_logits.append(action_logits)
            all_values.append(value)
            all_confidences.append(confidence)
        
        # Stack outputs
        action_logits_stack = torch.stack(all_action_logits, dim=0)  # [num_models, batch, 3]
        values_stack = torch.stack(all_values, dim=0)                # [num_models, batch, 1]
        confidences_stack = torch.stack(all_confidences, dim=0)      # [num_models, batch, 1]
        
        # Apply ensemble weights
        weights = F.softmax(self.ensemble_weights, dim=0).view(-1, 1, 1)
        
        # Weighted combination
        action_logits = (action_logits_stack * weights).sum(dim=0)
        state_value = (values_stack * weights).sum(dim=0)
        confidence = (confidences_stack * weights).sum(dim=0)
        
        return action_logits, state_value, confidence
    
    def get_ensemble_uncertainty(self, state: torch.Tensor) -> torch.Tensor:
        """Get uncertainty estimate from ensemble disagreement"""
        with torch.no_grad():
            all_action_probs = []
            
            for model in self.models:
                action_logits, _, _ = model(state)
                action_probs = F.softmax(action_logits, dim=-1)
                all_action_probs.append(action_probs)
            
            # Stack probabilities
            probs_stack = torch.stack(all_action_probs, dim=0)     # [num_models, batch, 3]
            
            # Compute variance across ensemble
            variance = torch.var(probs_stack, dim=0).mean(dim=-1)  # [batch]
            
            return variance


# Factory function to create policy networks
def create_policy_network(network_type: str, state_dim: int, config: dict = None):
    """Create policy network based on type"""
    if config is None:
        config = {}
    
    if network_type == "simple" or network_type == "mlp":
        hidden_dims = config.get("hidden_dims", [256, 128])
        dropout = config.get("dropout", 0.1)
        return SimpleMLPPolicy(state_dim, hidden_dims, dropout)
    
    elif network_type == "transformer":
        hidden_dim = config.get("hidden_dim", 256)
        num_heads = config.get("num_heads", 4)
        num_layers = config.get("num_layers", 2)
        dropout = config.get("dropout", 0.1)
        return TransformerPolicy(state_dim, hidden_dim, num_heads, num_layers, dropout)
    
    elif network_type == "attention":
        hidden_dim = config.get("hidden_dim", 256)
        num_heads = config.get("num_heads", 4)
        dropout = config.get("dropout", 0.1)
        return AttentionPolicy(state_dim, hidden_dim, num_heads, dropout)
    
    elif network_type == "residual":
        hidden_dim = config.get("hidden_dim", 256)
        num_blocks = config.get("num_blocks", 3)
        dropout = config.get("dropout", 0.1)
        return ResidualPolicy(state_dim, hidden_dim, num_blocks, dropout)
    
    elif network_type == "ensemble":
        num_models = config.get("num_models", 3)
        model_type = config.get("ensemble_model_type", "mlp")
        return EnsemblePolicy(state_dim, num_models, model_type)
    
    else:
        raise ValueError(f"Unknown network type: {network_type}")