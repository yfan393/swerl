"""
training/rollout_utils.py
==========================
Utilities for generating rollouts and computing advantages.
"""
import logging
from typing import List, Tuple
import torch

logger = logging.getLogger(__name__)

def generate_rollouts(
    model,
    input_ids: torch.Tensor,
    num_rollouts: int = 2,
    max_new_tokens: int = 512,
) -> List[str]:
    """
    Generate multiple rollouts from a model.
    
    Args:
        model: Language model
        input_ids: Input token IDs
        num_rollouts: Number of rollouts to generate
        max_new_tokens: Max tokens to generate
    
    Returns:
        List of generated texts
    """
    outputs = []
    for _ in range(num_rollouts):
        try:
            output = model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.7,
            )
            outputs.append(output)
        except Exception as e:
            logger.error(f"Generation failed: {e}")
    
    return outputs

def compute_advantages(
    rewards: List[float],
    gamma: float = 0.99,
) -> torch.Tensor:
    """
    Compute advantage estimates from rewards.
    
    Args:
        rewards: List of reward values
        gamma: Discount factor
    
    Returns:
        Advantage tensor
    """
    if not rewards:
        return torch.tensor([])
    
    # Simple advantage: reward - mean
    rewards_tensor = torch.tensor(rewards)
    mean_reward = rewards_tensor.mean()
    advantages = rewards_tensor - mean_reward
    
    return advantages

def compute_returns(
    rewards: List[float],
    gamma: float = 0.99,
) -> torch.Tensor:
    """
    Compute discounted returns from rewards.
    
    Args:
        rewards: List of rewards
        gamma: Discount factor
    
    Returns:
        Returns tensor
    """
    if not rewards:
        return torch.tensor([])
    
    returns = []
    cumulative = 0
    for r in reversed(rewards):
        cumulative = r + gamma * cumulative
        returns.insert(0, cumulative)
    
    return torch.tensor(returns)
