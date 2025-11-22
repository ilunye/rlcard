"""
DeltaDou Trainer
Implements the self-play training process as described in the DeltaDou paper.

Training Process:
1. Bootstrap: 200,000 games with heuristic algorithm, then supervised learning
2. Self-play episodes: Each episode contains configurable number of games
3. FPMCTS: 400 playouts per search
4. Inference: Used when any player has < 15 cards
5. Temperature scheduling: High temperature at start, gradually decreases
"""

import os
import numpy as np
import torch
import torch.nn.functional as F
from collections import deque
from tqdm import tqdm

import rlcard
from rlcard.agents import RandomAgent
from rlcard.utils import set_seed, tournament
from rlcard.agents.deltadou.deltadou_agent import DeltaDouAgent
from rlcard.agents.deltadou.policy_value_net import PolicyValueNet, create_policy_value_net
from rlcard.games.doudizhu.utils import ACTION_2_ID


class PolicyValueTrainer:
    """
    Trainer for Policy-Value Network
    
    Loss function: l = C_1(z - v)^2 - π^T log p + C_2||θ||_2
    """
    
    def __init__(self, model, learning_rate=0.001, value_loss_coef=1.0, 
                 l2_reg_coef=1e-4, device=None):
        """
        Args:
            model (PolicyValueNet): Policy-Value network
            learning_rate (float): Learning rate
            value_loss_coef (float): C_1 - scaling factor for value loss
            l2_reg_coef (float): C_2 - L2 regularization coefficient
            device: torch device
        """
        self.model = model
        self.value_loss_coef = value_loss_coef
        self.l2_reg_coef = l2_reg_coef
        self.device = device if device is not None else model.device
        
        # Optimizer
        self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=l2_reg_coef)
        
        # Loss functions
        self.value_loss_fn = torch.nn.MSELoss()
    
    def train_step(self, states, target_policies, target_values):
        """
        Perform one training step
        
        Args:
            states (torch.Tensor): Batch of states (batch, state_dim)
            target_policies (torch.Tensor): Target policy distributions (batch, num_actions)
            target_values (torch.Tensor): Target values (batch,)
            
        Returns:
            loss (float): Total loss
            value_loss (float): Value loss component
            policy_loss (float): Policy loss component
        """
        self.model.train()
        
        # Forward pass
        policy_logits, predicted_values = self.model(states)
        
        # Value loss: C_1 * (z - v)^2
        predicted_values = predicted_values.squeeze()  # (batch,)
        value_loss = self.value_loss_coef * self.value_loss_fn(predicted_values, target_values)
        
        # Policy loss: -π^T log p (cross-entropy)
        policy_loss = -torch.sum(target_policies * F.log_softmax(policy_logits, dim=1), dim=1).mean()
        
        # Total loss
        total_loss = value_loss + policy_loss
        
        # Backward pass
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()
        
        return total_loss.item(), value_loss.item(), policy_loss.item()
    
    def train_batch(self, batch_data):
        """
        Train on a batch of data
        
        Args:
            batch_data (list): List of (state, target_policy, target_value) tuples
            
        Returns:
            loss (float): Average loss
        """
        states = []
        target_policies = []
        target_values = []
        
        for state, policy, value in batch_data:
            states.append(state)
            target_policies.append(policy)
            target_values.append(value)
        
        # Convert to tensors
        states = torch.FloatTensor(np.array(states)).to(self.device)
        target_policies = torch.FloatTensor(np.array(target_policies)).to(self.device)
        target_values = torch.FloatTensor(np.array(target_values)).to(self.device)
        
        return self.train_step(states, target_policies, target_values)


class DeltaDouTrainer:
    """
    DeltaDou Self-play Trainer
    
    Implements the complete training pipeline:
    1. Bootstrap phase with heuristic algorithm
    2. Self-play episodes with FPMCTS
    3. Network training from collected data
    """
    
    def __init__(self, 
                 num_games_per_episode=8000,
                 bootstrap_games=200000,
                 fpmcts_simulations=400,
                 inference_threshold=15,
                 temperature_start=1.0,
                 temperature_end=0.1,
                 temperature_decay=0.95,
                 batch_size=32,
                 learning_rate=0.001,
                 value_loss_coef=1.0,
                 l2_reg_coef=1e-4,
                 state_dim=790,
                 num_actions=54,
                 num_residual_blocks=10,
                 base_channels=128,
                 device=None,
                 save_dir='experiments/deltadou_result/',
                 save_every=10,
                 evaluate_every=5,
                 num_eval_games=1000):
        """
        Args:
            num_games_per_episode (int): Number of games per self-play episode (default: 8000)
            bootstrap_games (int): Number of bootstrap games with heuristic (default: 200000)
            fpmcts_simulations (int): Number of FPMCTS simulations per search (default: 400)
            inference_threshold (int): Use inference when player has < this many cards (default: 15)
            temperature_start (float): Starting temperature for exploration (default: 1.0)
            temperature_end (float): Ending temperature (default: 0.1)
            temperature_decay (float): Temperature decay factor per episode (default: 0.95)
            batch_size (int): Training batch size
            learning_rate (float): Learning rate
            value_loss_coef (float): Value loss coefficient
            l2_reg_coef (float): L2 regularization coefficient
            state_dim (int): State dimension (790 for landlord, 901 for peasants)
            num_actions (int): Number of actions
            num_residual_blocks (int): Number of residual blocks in network
            base_channels (int): Base channels in network
            device: torch device
            save_dir (str): Directory to save models
            save_every (int): Save model every N episodes
            evaluate_every (int): Evaluate every N episodes
            num_eval_games (int): Number of games for evaluation
        """
        self.num_games_per_episode = num_games_per_episode
        self.bootstrap_games = bootstrap_games
        self.fpmcts_simulations = fpmcts_simulations
        self.inference_threshold = inference_threshold
        self.temperature_start = temperature_start
        self.temperature_end = temperature_end
        self.temperature_decay = temperature_decay
        self.batch_size = batch_size
        self.save_dir = save_dir
        self.save_every = save_every
        self.evaluate_every = evaluate_every
        self.num_eval_games = num_eval_games
        
        # Create save directory
        os.makedirs(save_dir, exist_ok=True)
        
        # Device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
        
        # Create environment
        self.env = rlcard.make('doudizhu', config={'seed': 42, 'allow_step_back': True})
        
        # Create agents for each position (landlord, peasant_up, peasant_down)
        # Each position has its own network
        self.agents = []
        self.networks = []
        self.trainers = []
        
        # State dimensions for each position: [landlord, peasant_up, peasant_down]
        state_dims = [790, 901, 901]
        
        for pos in range(3):
            # Create network
            network = create_policy_value_net(
                state_dim=state_dims[pos],
                num_actions=num_actions,
                num_residual_blocks=num_residual_blocks,
                base_channels=base_channels,
                device=self.device
            )
            self.networks.append(network)
            
            # Create agent
            agent = DeltaDouAgent(
                num_actions=num_actions,
                simulate_num=fpmcts_simulations,
                state_dim=state_dims[pos],
                num_residual_blocks=num_residual_blocks,
                base_channels=base_channels,
                device=self.device
            )
            # Replace agent's network with the created network
            agent.policy_value_net = network
            self.agents.append(agent)
            
            # Create trainer
            trainer = PolicyValueTrainer(
                network,
                learning_rate=learning_rate,
                value_loss_coef=value_loss_coef,
                l2_reg_coef=l2_reg_coef,
                device=self.device
            )
            self.trainers.append(trainer)
        
        # Reservoir for storing training data
        self.reservoir = [deque(maxlen=1000000) for _ in range(3)]  # One for each position
        
        # Training statistics
        self.current_episode = 0
        self.current_temperature = temperature_start
        
    def _get_temperature(self, episode):
        """Get temperature for current episode"""
        temp = self.temperature_start * (self.temperature_decay ** episode)
        return max(temp, self.temperature_end)
    
    def _should_use_inference(self, state):
        """Check if inference should be used (when player has < threshold cards)"""
        raw_obs = state.get('raw_obs', state)
        current_hand = raw_obs.get('current_hand', '')
        num_cards = len(current_hand)
        return num_cards < self.inference_threshold
    
    def _sample_action_with_temperature(self, probs, temperature):
        """Sample action with temperature"""
        if temperature == 0:
            return np.argmax(probs)
        
        # Apply temperature
        log_probs = np.log(probs + 1e-8)
        scaled_log_probs = log_probs / temperature
        exp_probs = np.exp(scaled_log_probs)
        exp_probs = exp_probs / exp_probs.sum()
        
        return np.random.choice(len(exp_probs), p=exp_probs)
    
    def _collect_training_data(self, trajectories, payoffs):
        """
        Collect training data from self-play game
        
        Args:
            trajectories (list): Game trajectories for each player
            payoffs (list): Final payoffs for each player
        """
        for player_id in range(3):
            player_trajectory = trajectories[player_id]
            payoff = payoffs[player_id]
            
            # Extract (state, action, next_state) pairs
            for i in range(0, len(player_trajectory) - 2, 2):
                state = player_trajectory[i]
                action = player_trajectory[i + 1]
                
                # Get state vector
                state_vector = self.agents[player_id]._extract_state_vector(state)
                
                # Get FPMCTS policy (if available) or use network policy
                # For training data, we want the policy from FPMCTS
                try:
                    # Try to get FPMCTS policy
                    env = state.get('_env', None)
                    if env is not None and hasattr(env, 'game'):
                        perfect_info = env.get_perfect_information()
                        raw_obs = state.get('raw_obs', state)
                        info_state = state
                        game = env.game
                        
                        # Perform FPMCTS search
                        action_probs = self.agents[player_id]._search(
                            info_state, perfect_info, player_id, game
                        )
                        
                        # Convert to action ID probabilities
                        legal_actions = state.get('raw_legal_actions', [])
                        target_policy = np.zeros(self.agents[player_id].num_actions)
                        
                        for action_str, prob in action_probs.items():
                            if action_str in ACTION_2_ID:
                                action_id = ACTION_2_ID[action_str]
                                target_policy[action_id] = prob
                    else:
                        # Fallback to network policy
                        legal_mask = np.zeros(self.agents[player_id].num_actions)
                        legal_actions = state.get('raw_legal_actions', [])
                        for action_str in legal_actions:
                            if action_str in ACTION_2_ID:
                                legal_mask[ACTION_2_ID[action_str]] = 1.0
                        
                        policy_probs, _ = self.networks[player_id].predict(state_vector, legal_mask)
                        target_policy = policy_probs
                except:
                    # Fallback: uniform over legal actions
                    target_policy = np.zeros(self.agents[player_id].num_actions)
                    legal_action_ids = list(state['legal_actions'].keys())
                    if legal_action_ids:
                        uniform_prob = 1.0 / len(legal_action_ids)
                        for aid in legal_action_ids:
                            target_policy[aid] = uniform_prob
                
                # Store in reservoir
                self.reservoir[player_id].append((state_vector, target_policy, payoff))
    
    def _train_networks(self):
        """Train networks on data from reservoir"""
        for player_id in range(3):
            if len(self.reservoir[player_id]) < self.batch_size:
                continue
            
            # Sample batch
            batch_indices = np.random.choice(
                len(self.reservoir[player_id]), 
                size=min(self.batch_size, len(self.reservoir[player_id])),
                replace=False
            )
            batch_data = [self.reservoir[player_id][i] for i in batch_indices]
            
            # Train
            loss, value_loss, policy_loss = self.trainers[player_id].train_batch(batch_data)
            
            if self.current_episode % 100 == 0:
                print(f"Player {player_id} - Loss: {loss:.4f}, Value Loss: {value_loss:.4f}, Policy Loss: {policy_loss:.4f}")
    
    def bootstrap_phase(self):
        """Bootstrap phase: Use heuristic algorithm for initial training"""
        print("=" * 50)
        print("Bootstrap Phase: Generating initial training data")
        print("=" * 50)
        
        # Use random agents as heuristic (can be replaced with better heuristic)
        heuristic_agents = [RandomAgent(num_actions=self.env.num_actions) for _ in range(3)]
        self.env.set_agents(heuristic_agents)
        
        # Collect games
        for game_idx in tqdm(range(self.bootstrap_games), desc="Bootstrap games"):
            trajectories, payoffs = self.env.run(is_training=True)
            self._collect_training_data(trajectories, payoffs)
        
        print(f"Collected {sum(len(r) for r in self.reservoir)} training samples")
        
        # Train initial networks
        print("Training initial networks...")
        for epoch in tqdm(range(100), desc="Initial training"):
            self._train_networks()
        
        # Save initial models
        self.save_models(episode=0)
        print("Bootstrap phase completed!")
    
    def self_play_episode(self, episode):
        """Run one self-play episode"""
        # Update temperature
        self.current_temperature = self._get_temperature(episode)
        
        # Set agents
        self.env.set_agents(self.agents)
        
        # Collect games
        for game_idx in tqdm(range(self.num_games_per_episode), 
                            desc=f"Episode {episode} - Self-play"):
            trajectories, payoffs = self.env.run(is_training=True)
            self._collect_training_data(trajectories, payoffs)
        
        # Train networks
        print(f"Training networks for episode {episode}...")
        for _ in range(10):  # Train for 10 iterations per episode
            self._train_networks()
    
    def evaluate(self):
        """Evaluate current agents"""
        print("Evaluating agents...")
        
        # Create evaluation agents (current best)
        eval_agents = []
        for agent in self.agents:
            eval_agent = DeltaDouAgent(
                num_actions=self.env.num_actions,
                simulate_num=self.fpmcts_simulations,
                state_dim=agent.state_dim,
                device=self.device
            )
            eval_agent.policy_value_net = agent.policy_value_net
            eval_agents.append(eval_agent)
        
        # Play against random agents
        random_agents = [RandomAgent(num_actions=self.env.num_actions) for _ in range(3)]
        
        # Test as landlord
        test_agents = [eval_agents[0], random_agents[1], random_agents[2]]
        self.env.set_agents(test_agents)
        landlord_rewards = tournament(self.env, self.num_eval_games)
        
        # Test as peasants
        test_agents = [random_agents[0], eval_agents[1], eval_agents[2]]
        self.env.set_agents(test_agents)
        peasant_rewards = tournament(self.env, self.num_eval_games)
        
        print(f"Landlord win rate: {landlord_rewards[0]:.4f}")
        print(f"Peasant win rate: {(peasant_rewards[1] + peasant_rewards[2]) / 2:.4f}")
        
        return {
            'landlord_reward': landlord_rewards[0],
            'peasant_reward': (peasant_rewards[1] + peasant_rewards[2]) / 2
        }
    
    def save_models(self, episode):
        """Save models"""
        for player_id in range(3):
            model_path = os.path.join(self.save_dir, f'model_player_{player_id}_episode_{episode}.pth')
            torch.save({
                'model_state_dict': self.networks[player_id].state_dict(),
                'episode': episode,
                'state_dim': self.agents[player_id].state_dim,
                'num_actions': self.env.num_actions
            }, model_path)
        print(f"Models saved for episode {episode}")
    
    def load_models(self, episode):
        """Load models"""
        for player_id in range(3):
            model_path = os.path.join(self.save_dir, f'model_player_{player_id}_episode_{episode}.pth')
            if os.path.exists(model_path):
                checkpoint = torch.load(model_path, map_location=self.device)
                self.networks[player_id].load_state_dict(checkpoint['model_state_dict'])
                print(f"Loaded model for player {player_id} from episode {episode}")
    
    def start(self, num_episodes=100, start_episode=0, skip_bootstrap=False):
        """
        Start training
        
        Args:
            num_episodes (int): Number of self-play episodes
            start_episode (int): Starting episode (for resuming)
            skip_bootstrap (bool): Skip bootstrap phase if True
        """
        print("=" * 50)
        print("DeltaDou Training Started")
        print("=" * 50)
        print(f"Device: {self.device}")
        print(f"Games per episode: {self.num_games_per_episode}")
        print(f"FPMCTS simulations: {self.fpmcts_simulations}")
        print(f"Total episodes: {num_episodes}")
        print("=" * 50)
        
        # Bootstrap phase
        if not skip_bootstrap and start_episode == 0:
            self.bootstrap_phase()
        elif start_episode > 0:
            self.load_models(start_episode - 1)
        
        # Self-play episodes
        for episode in range(start_episode, start_episode + num_episodes):
            self.current_episode = episode
            print(f"\n{'=' * 50}")
            print(f"Episode {episode + 1}/{start_episode + num_episodes}")
            print(f"Temperature: {self.current_temperature:.4f}")
            print(f"{'=' * 50}")
            
            # Self-play
            self.self_play_episode(episode)
            
            # Evaluate
            if (episode + 1) % self.evaluate_every == 0:
                eval_results = self.evaluate()
            
            # Save
            if (episode + 1) % self.save_every == 0:
                self.save_models(episode + 1)
        
        print("\n" + "=" * 50)
        print("Training Completed!")
        print("=" * 50)

