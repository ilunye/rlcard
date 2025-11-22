"""
Policy-Value Network for DeltaDou (FPMCTS)
Based on the paper: "DeltaDou: Mastering DouDizhu with Self-Play Deep Reinforcement Learning"

Network Architecture:
- Input: State encoding (condensed representation)
- 10 Residual blocks with 1D convolutions, batch normalization, and ReLU
- Output: Policy (rank and category) + Value estimate
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import OrderedDict


class ResidualBlock1D(nn.Module):
    """1D Convolutional Residual Block with Batch Normalization"""
    
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(ResidualBlock1D, self).__init__()
        
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, stride, padding)
        self.bn2 = nn.BatchNorm1d(out_channels)
        
        # Shortcut connection
        self.shortcut = nn.Sequential()
        if in_channels != out_channels or stride != 1:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, 1, stride),
                nn.BatchNorm1d(out_channels)
            )
    
    def forward(self, x):
        residual = self.shortcut(x)
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        out = F.relu(out)
        return out


class PolicyValueNet(nn.Module):
    """
    Policy-Value Network for Doudizhu
    
    Architecture:
    - Input projection layer
    - 10 Residual blocks (1D convolutions)
    - Policy head: outputs action probabilities (rank and category encoding)
    - Value head: outputs state value estimate
    """
    
    def __init__(self, state_dim=790, num_actions=54, num_residual_blocks=10, 
                 base_channels=128, device=None):
        """
        Args:
            state_dim (int): Dimension of input state (default: 790 for landlord, 901 for peasants)
            num_actions (int): Number of action categories (default: 54 for abstracted actions)
            num_residual_blocks (int): Number of residual blocks (default: 10)
            base_channels (int): Base number of channels for convolutions
            device: torch device
        """
        super(PolicyValueNet, self).__init__()
        
        self.state_dim = state_dim
        self.num_actions = num_actions
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Input projection: convert state to feature map
        # Reshape state to (batch, channels, length) for 1D convolution
        # We'll treat the state as a 1D sequence
        self.input_proj = nn.Sequential(
            nn.Linear(state_dim, base_channels * 8),
            nn.BatchNorm1d(base_channels * 8),
            nn.ReLU()
        )
        
        # Reshape for 1D convolution: (batch, channels, length)
        # We'll use a fixed length representation
        self.conv_input = nn.Conv1d(base_channels * 8, base_channels, kernel_size=1)
        
        # Residual blocks
        self.residual_blocks = nn.ModuleList([
            ResidualBlock1D(base_channels, base_channels) 
            for _ in range(num_residual_blocks)
        ])
        
        # Policy head: outputs action probabilities
        # The output encodes rank and category (not full action details)
        self.policy_head = nn.Sequential(
            nn.Conv1d(base_channels, base_channels // 2, kernel_size=1),
            nn.BatchNorm1d(base_channels // 2),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),  # Global average pooling
            nn.Flatten(),
            nn.Linear(base_channels // 2, num_actions)
        )
        
        # Value head: outputs state value
        self.value_head = nn.Sequential(
            nn.Conv1d(base_channels, base_channels // 2, kernel_size=1),
            nn.BatchNorm1d(base_channels // 2),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),  # Global average pooling
            nn.Flatten(),
            nn.Linear(base_channels // 2, base_channels // 4),
            nn.ReLU(),
            nn.Linear(base_channels // 4, 1),
            nn.Tanh()  # Value is normalized to [-1, 1]
        )
        
        self.to(self.device)
    
    def forward(self, state):
        """
        Forward pass
        
        Args:
            state (torch.Tensor): State tensor of shape (batch, state_dim)
            
        Returns:
            policy (torch.Tensor): Action probabilities of shape (batch, num_actions)
            value (torch.Tensor): State value of shape (batch, 1)
        """
        batch_size = state.size(0)
        
        # Project input
        x = self.input_proj(state)  # (batch, base_channels * 8)
        
        # Reshape for 1D convolution
        # We'll treat it as a sequence of length 1 with many channels
        x = x.unsqueeze(2)  # (batch, base_channels * 8, 1)
        x = self.conv_input(x)  # (batch, base_channels, 1)
        
        # Apply residual blocks
        for block in self.residual_blocks:
            x = block(x)
        
        # Policy and value heads
        policy_logits = self.policy_head(x)  # (batch, num_actions)
        value = self.value_head(x)  # (batch, 1)
        
        return policy_logits, value
    
    def predict(self, state, legal_actions_mask=None):
        """
        Predict action probabilities and value
        
        Args:
            state (np.ndarray or torch.Tensor): State array
            legal_actions_mask (np.ndarray): Mask for legal actions (optional)
            
        Returns:
            policy_probs (np.ndarray): Action probabilities
            value (float): State value estimate
        """
        self.eval()
        
        if isinstance(state, np.ndarray):
            state = torch.FloatTensor(state).to(self.device)
        
        if len(state.shape) == 1:
            state = state.unsqueeze(0)  # Add batch dimension
        
        with torch.no_grad():
            policy_logits, value = self.forward(state)
            policy_probs = F.softmax(policy_logits, dim=1)
            
            # Apply legal actions mask if provided
            if legal_actions_mask is not None:
                if isinstance(legal_actions_mask, np.ndarray):
                    legal_actions_mask = torch.FloatTensor(legal_actions_mask).to(self.device)
                policy_probs = policy_probs * legal_actions_mask
                policy_probs = policy_probs / (policy_probs.sum(dim=1, keepdim=True) + 1e-8)
        
        policy_probs = policy_probs.cpu().numpy()
        value = value.cpu().numpy()
        
        return policy_probs[0] if policy_probs.shape[0] == 1 else policy_probs, value[0, 0] if value.shape[0] == 1 else value
    
    def get_policy(self, state, legal_actions_mask=None):
        """
        Get policy distribution over actions
        
        Args:
            state (np.ndarray): State array
            legal_actions_mask (np.ndarray): Mask for legal actions (optional)
            
        Returns:
            policy_probs (dict): Dictionary mapping action indices to probabilities
        """
        policy_probs, _ = self.predict(state, legal_actions_mask)
        
        if isinstance(policy_probs, np.ndarray) and len(policy_probs.shape) == 1:
            policy_dict = {i: float(prob) for i, prob in enumerate(policy_probs)}
        else:
            policy_dict = {}
            for i, prob in enumerate(policy_probs[0]):
                policy_dict[i] = float(prob)
        
        return policy_dict
    
    def get_value(self, state):
        """
        Get value estimate for state
        
        Args:
            state (np.ndarray): State array
            
        Returns:
            value (float): State value estimate
        """
        _, value = self.predict(state)
        return float(value)


def create_policy_value_net(state_dim=790, num_actions=54, num_residual_blocks=10, 
                           base_channels=128, device=None):
    """
    Factory function to create a Policy-Value Network
    
    Args:
        state_dim (int): Dimension of input state
        num_actions (int): Number of action categories
        num_residual_blocks (int): Number of residual blocks
        base_channels (int): Base number of channels
        device: torch device
        
    Returns:
        model (PolicyValueNet): Policy-Value network
    """
    model = PolicyValueNet(
        state_dim=state_dim,
        num_actions=num_actions,
        num_residual_blocks=num_residual_blocks,
        base_channels=base_channels,
        device=device
    )
    return model

