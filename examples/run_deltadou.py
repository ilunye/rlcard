"""
Example script for training DeltaDou agent
"""

import os
import argparse
import torch

from rlcard.agents.deltadou.trainer import DeltaDouTrainer
from rlcard.utils import get_device


def train(args):
    """Train DeltaDou agent"""
    
    # Get device
    device = get_device()
    print(f"Using device: {device}")
    
    # Create trainer
    trainer = DeltaDouTrainer(
        num_games_per_episode=args.num_games_per_episode,
        bootstrap_games=args.bootstrap_games,
        fpmcts_simulations=args.fpmcts_simulations,
        inference_threshold=args.inference_threshold,
        temperature_start=args.temperature_start,
        temperature_end=args.temperature_end,
        temperature_decay=args.temperature_decay,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        value_loss_coef=args.value_loss_coef,
        l2_reg_coef=args.l2_reg_coef,
        num_residual_blocks=args.num_residual_blocks,
        base_channels=args.base_channels,
        device=device,
        save_dir=args.save_dir,
        save_every=args.save_every,
        evaluate_every=args.evaluate_every,
        num_eval_games=args.num_eval_games
    )
    
    # Start training
    trainer.start(
        num_episodes=args.num_episodes,
        start_episode=args.start_episode,
        skip_bootstrap=args.skip_bootstrap,
        force_bootstrap=args.force_bootstrap
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser("DeltaDou Training")
    
    # Training parameters
    parser.add_argument('--num_games_per_episode', type=int, default=8000,
                        help='Number of games per self-play episode (default: 8000)')
    parser.add_argument('--num_episodes', type=int, default=100,
                        help='Number of self-play episodes (default: 100)')
    parser.add_argument('--bootstrap_games', type=int, default=21000,
                        help='Number of bootstrap games with heuristic (default: 21000)')
    parser.add_argument('--fpmcts_simulations', type=int, default=400,
                        help='Number of FPMCTS simulations per search (default: 400)')
    parser.add_argument('--inference_threshold', type=int, default=15,
                        help='Use inference when player has < this many cards (default: 15)')
    
    # Temperature scheduling
    parser.add_argument('--temperature_start', type=float, default=1.0,
                        help='Starting temperature (default: 1.0)')
    parser.add_argument('--temperature_end', type=float, default=0.1,
                        help='Ending temperature (default: 0.1)')
    parser.add_argument('--temperature_decay', type=float, default=0.95,
                        help='Temperature decay factor per episode (default: 0.95)')
    
    # Network parameters
    parser.add_argument('--num_residual_blocks', type=int, default=10,
                        help='Number of residual blocks (default: 10)')
    parser.add_argument('--base_channels', type=int, default=128,
                        help='Base number of channels (default: 128)')
    
    # Training hyperparameters
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Training batch size (default: 32)')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='Learning rate (default: 0.001)')
    parser.add_argument('--value_loss_coef', type=float, default=1.0,
                        help='Value loss coefficient (default: 1.0)')
    parser.add_argument('--l2_reg_coef', type=float, default=1e-4,
                        help='L2 regularization coefficient (default: 1e-4)')
    
    # Save and evaluation
    parser.add_argument('--save_dir', type=str, default='experiments/deltadou_result/',
                        help='Directory to save models (default: experiments/deltadou_result/)')
    parser.add_argument('--save_every', type=int, default=10,
                        help='Save model every N episodes (default: 10)')
    parser.add_argument('--evaluate_every', type=int, default=5,
                        help='Evaluate every N episodes (default: 5)')
    parser.add_argument('--num_eval_games', type=int, default=1000,
                        help='Number of games for evaluation (default: 1000)')
    
    # Resume training
    parser.add_argument('--start_episode', type=int, default=0,
                        help='Starting episode for resuming training (default: 0)')
    parser.add_argument('--skip_bootstrap', action='store_true',
                        help='Skip bootstrap phase')
    parser.add_argument('--force_bootstrap', action='store_true',
                        help='Force regenerate bootstrap data even if saved data exists')
    
    args = parser.parse_args()
    
    # Train
    train(args)

