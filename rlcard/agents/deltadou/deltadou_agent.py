import numpy as np
import copy
from collections import defaultdict
from rlcard.games.doudizhu import Game
from rlcard.games.doudizhu.utils import ACTION_2_ID, ID_2_ACTION, cards2str, CARD_RANK_STR
from rlcard.agents.deltadou.policy_value_net import PolicyValueNet, create_policy_value_net


class MCTSNode:
    ''' MCTS Node for FPMCTS
    '''
    def __init__(self, info_state, player_id):
        self.info_state = info_state  # Information set (u^i)
        self.player_id = player_id
        self.visits = 0  # N(u^i)
        self.Q = defaultdict(float)  # Q(u^i, a) - average Q-value for each action
        self.N = defaultdict(int)  # N(u^i, a) - visit count for each action
        self.children = {}  # child nodes: {action: MCTSNode}
        self.is_decision_node = True  # True for decision node, False for chance node
        self.prior_policy = {}  # π^i(u^i, a) - prior policy from network (placeholder for now)
        
    def is_leaf(self):
        return len(self.children) == 0
    
    def get_legal_actions(self):
        ''' Get legal actions from info state
        '''
        if 'actions' in self.info_state:
            return self.info_state['actions']
        return []


class DeltaDouAgent(object):
    ''' FPMCTS agent for Doudizhu based on the DeltaDou paper
    '''
    
    def __init__(self, num_actions, simulate_num=100, c_puct=1.0, 
                 state_dim=790, num_residual_blocks=10, base_channels=128,
                 model_path=None, device=None):
        ''' Initialize the FPMCTS agent
        
        Args:
            num_actions (int): The size of the output action space
            simulate_num (int): Number of simulations per search
            c_puct (float): Exploration constant for p-UCT
            state_dim (int): Dimension of state (790 for landlord, 901 for peasants)
            num_residual_blocks (int): Number of residual blocks in network
            base_channels (int): Base number of channels in network
            model_path (str): Path to load pretrained model (optional)
            device: torch device (optional)
        '''
        self.use_raw = False
        self.num_actions = num_actions
        self.simulate_num = simulate_num
        self.c_puct = c_puct
        self.state_dim = state_dim
        
        # Search trees for each player: {player_id: {info_state_hash: MCTSNode}}
        self.search_trees = defaultdict(dict)
        
        # Track printed fallbacks to avoid spam
        self._printed_fallbacks = set()
        
        # Initialize Policy-Value Network
        self.policy_value_net = create_policy_value_net(
            state_dim=state_dim,
            num_actions=num_actions,
            num_residual_blocks=num_residual_blocks,
            base_channels=base_channels,
            device=device
        )
        
        # Load pretrained model if provided
        if model_path is not None:
            self.load_model(model_path)
        
        # Trainer for the network
        self.trainer = None  # Will be initialized when needed
        
    def _hash_info_state(self, info_state):
        ''' Hash information state to use as key in search tree
        '''
        # Use a combination of current hand, trace, and legal actions
        key_parts = [
            info_state.get('current_hand', ''),
            str(info_state.get('trace', [])),
            str(sorted(info_state.get('actions', [])))
        ]
        return hash(tuple(key_parts))
    
    def _determinize(self, info_state, perfect_info):
        ''' Determinization: create a deterministic world from information state
        Currently uses random assignment of unknown cards
        
        Args:
            info_state (dict): Current information state
            perfect_info (dict): Perfect information from environment
            
        Returns:
            dict: Deterministic state with all cards assigned
        '''
        # For now, we'll use the perfect information directly
        # In a more sophisticated implementation, we would sample from
        # a distribution based on game history
        deterministic_state = copy.deepcopy(perfect_info)
        return deterministic_state
    
    def _extract_state_vector(self, state):
        ''' Extract state vector from state dict for network input
        
        Args:
            state (dict): State dictionary (from env or info_state)
            
        Returns:
            np.ndarray: State vector
        '''
        # Try to get obs from state (from env._extract_state)
        if 'obs' in state:
            return state['obs']
        
        # If state is from game.get_state(), we need to construct it
        # This is a simplified version - proper implementation would use env._extract_state
        # For now, return zero vector as fallback
        self._print_fallback_once("[Fallback] _extract_state_vector: No 'obs' key in state, returning zero vector")
        return np.zeros(self.state_dim)
    
    def _print_fallback_once(self, message):
        """Print fallback message only once"""
        if message not in self._printed_fallbacks:
            print(message)
            self._printed_fallbacks.add(message)
    
    def _get_prior_policy(self, info_state, legal_actions):
        ''' Get prior policy π^i(u^i, a) from network
        
        Args:
            info_state (dict): Current information state
            legal_actions (list): List of legal actions (action strings)
            
        Returns:
            dict: {action: probability}
        '''
        # Convert info_state to network input format
        state_vector = self._extract_state_vector(info_state)
        
        # Create legal actions mask
        # Convert action strings to action IDs
        legal_action_ids = []
        for action_str in legal_actions:
            if action_str in ACTION_2_ID:
                legal_action_ids.append(ACTION_2_ID[action_str])
        
        # Create mask
        legal_mask = np.zeros(self.num_actions)
        if legal_action_ids:
            legal_mask[legal_action_ids] = 1.0
        
        # Get policy from network
        try:
            policy_probs = self.policy_value_net.get_policy(state_vector, legal_mask)
            
            # Convert back to action strings
            action_probs = {}
            for action_str in legal_actions:
                if action_str in ACTION_2_ID:
                    action_id = ACTION_2_ID[action_str]
                    action_probs[action_str] = policy_probs.get(action_id, 0.0)
                else:
                    action_probs[action_str] = 0.0
            
            # Normalize
            total_prob = sum(action_probs.values())
            if total_prob > 0:
                action_probs = {k: v / total_prob for k, v in action_probs.items()}
            else:
                # Fallback to uniform
                self._print_fallback_once("[Fallback] _get_prior_policy: Total probability is 0, using uniform distribution")
                uniform_prob = 1.0 / len(legal_actions) if legal_actions else 0.0
                action_probs = {action: uniform_prob for action in legal_actions}
            
            return action_probs
        except Exception as e:
            # Fallback to uniform if network fails
            self._print_fallback_once(f"[Fallback] _get_prior_policy: Network failed with exception: {e}, using uniform distribution")
            num_legal = len(legal_actions)
            if num_legal == 0:
                return {}
            uniform_prob = 1.0 / num_legal
            return {action: uniform_prob for action in legal_actions}
    
    def _get_value(self, info_state):
        ''' Get value estimate v^i(u^i) from network
        
        Args:
            info_state (dict): Current information state
            
        Returns:
            float: Value estimate
        '''
        # Convert info_state to network input format
        state_vector = self._extract_state_vector(info_state)
        
        # Get value from network
        try:
            value = self.policy_value_net.get_value(state_vector)
            return value
        except Exception as e:
            # Fallback to 0 if network fails
            self._print_fallback_once(f"[Fallback] _get_value: Network failed with exception: {e}, returning 0.0")
            return 0.0
    
    def _puct_select(self, node, legal_actions):
        ''' Select action using p-UCT formula
        
        π_tree(u^i) = arg max_{a ∈ A(u^i)} [Q(u^i, a) + π^i(u^i, a) * √(N(u^i)) / (1 + N(u^i, a))]
        
        Args:
            node (MCTSNode): Current node
            legal_actions (list): List of legal actions
            
        Returns:
            str: Selected action
        '''
        if not legal_actions:
            return None
        
        best_action = None
        best_score = float('-inf')
        
        # Get prior policy if not already computed
        if not node.prior_policy:
            node.prior_policy = self._get_prior_policy(node.info_state, legal_actions)
        
        N_u = node.visits
        
        for action in legal_actions:
            Q_ua = node.Q[action]
            pi_ua = node.prior_policy.get(action, 0.0)
            N_ua = node.N[action]
            
            # p-UCT formula
            if N_ua == 0:
                # First visit: use prior only
                score = Q_ua + pi_ua * np.sqrt(N_u + 1)
            else:
                score = Q_ua + pi_ua * np.sqrt(N_u) / (1 + N_ua)
            
            if score > best_score:
                best_score = score
                best_action = action
        
        return best_action
    
    def _is_terminal(self, game_state):
        ''' Check if game state is terminal
        
        Args:
            game_state: Game state from game.get_state()
            
        Returns:
            bool: True if terminal
        '''
        # Check if game is over (this would need to be checked from the game)
        # For now, we'll assume the game object tracks this
        return False  # Placeholder
    
    def _get_reward(self, game, player_id):
        ''' Get reward for player at terminal state
        
        Args:
            game: Game object
            player_id (int): Player ID
            
        Returns:
            float: Reward for the player
        '''
        if not game.is_over():
            return 0.0
        
        payoffs = game.judger.judge_payoffs(game.round.landlord_id, game.winner_id)
        return payoffs[player_id]
    
    def _expand_tree(self, game, node, deterministic_state, action):
        ''' Expand tree by two levels (current player + two opponents)
        
        Args:
            game: Game object
            node (MCTSNode): Current node to expand
            deterministic_state: Deterministic state
            action: Action taken at current node
            
        Returns:
            dict: New information state after expansion, or None if terminal
        '''
        # Step 1: Current player takes action
        if game.is_over():
            return None
        
        current_state, next_player = game.step(action)
        
        # Step 2: Expand two levels (two opponent moves)
        for i in range(2):
            if game.is_over():
                break
            
            # Get opponent's information state
            opponent_id = game.round.current_player
            opponent_state = game.get_state(opponent_id)
            
            # Sample action from opponent's policy (placeholder: random)
            opponent_legal_actions = opponent_state.get('actions', [])
            if not opponent_legal_actions:
                break
            
            # TODO: Sample from opponent policy network
            opponent_action = np.random.choice(opponent_legal_actions)
            
            # Take opponent action
            next_state, next_player = game.step(opponent_action)
            
            if game.is_over():
                break
        
        # Get new information state for current player
        if not game.is_over():
            new_player_id = game.round.current_player
            # Check if it's back to our player
            if new_player_id == node.player_id:
                new_info_state = game.get_state(new_player_id)
                return new_info_state
            else:
                # It's still an opponent's turn, return None (chance node)
                return None
        else:
            return None
    
    def _simulate(self, game, player_id, deterministic_state, initial_trace_length):
        ''' Simulate one path from root to leaf
        
        Args:
            game: Game object (will be modified during simulation)
            player_id (int): Current player ID
            deterministic_state: Deterministic state
            initial_trace_length: Initial trace length to restore state later
            
        Returns:
            float: Value estimate
        '''
        # Traverse tree until leaf
        current_info_state = game.get_state(player_id)
        path = []  # Store (node, action) pairs for backpropagation
        steps_taken = 0
        
        while not game.is_over():
            # Hash current information state
            info_hash = self._hash_info_state(current_info_state)
            
            # Get or create node in search tree
            if info_hash not in self.search_trees[player_id]:
                node = MCTSNode(current_info_state, player_id)
                self.search_trees[player_id][info_hash] = node
            else:
                node = self.search_trees[player_id][info_hash]
            
            # Check if terminal
            if game.is_over():
                reward = self._get_reward(game, player_id)
                # Backpropagate
                for n, a in path:
                    n.visits += 1
                    n.N[a] += 1
                    n.Q[a] += (reward - n.Q[a]) / n.N[a]
                return reward
            
            # Check if leaf
            legal_actions = current_info_state.get('actions', [])
            if not legal_actions:
                break
            
            # Check if we should expand (leaf node that hasn't been fully explored)
            if node.is_leaf() or node.visits == 0:
                # Select action using p-UCT
                action = self._puct_select(node, legal_actions)
                if action is None:
                    break
                
                # Expand tree (two levels)
                new_info_state = self._expand_tree(game, node, deterministic_state, action)
                steps_taken += 3  # Current player + 2 opponents
                
                # Add child node
                if new_info_state and not game.is_over():
                    child_hash = self._hash_info_state(new_info_state)
                    if child_hash not in self.search_trees[player_id]:
                        child_node = MCTSNode(new_info_state, player_id)
                        self.search_trees[player_id][child_hash] = child_node
                    node.children[action] = child_hash
                
                # Get value estimate
                if game.is_over():
                    value = self._get_reward(game, player_id)
                else:
                    value = self._get_value(new_info_state if new_info_state else current_info_state)
                
                # Backpropagate
                node.visits += 1
                node.N[action] += 1
                if node.N[action] > 0:
                    node.Q[action] += (value - node.Q[action]) / node.N[action]
                
                for n, a in reversed(path):
                    n.visits += 1
                    n.N[a] += 1
                    if n.N[a] > 0:
                        n.Q[a] += (value - n.Q[a]) / n.N[a]
                
                return value
            else:
                # Select action using p-UCT
                action = self._puct_select(node, legal_actions)
                if action is None:
                    break
                
                # Take action
                path.append((node, action))
                next_state, next_player = game.step(action)
                steps_taken += 1
                
                # Update current info state
                if not game.is_over():
                    current_info_state = game.get_state(player_id)
                else:
                    break
        
        # If we reach here, use value network estimate
        if current_info_state:
            value = self._get_value(current_info_state)
        else:
            value = 0.0
        
        # Backpropagate
        for n, a in reversed(path):
            n.visits += 1
            n.N[a] += 1
            if n.N[a] > 0:
                n.Q[a] += (value - n.Q[a]) / n.N[a]
        
        return value
    
    def _search(self, info_state, perfect_info, player_id, game):
        ''' Main search function: perform multiple simulations
        
        Args:
            info_state (dict): Current information state
            perfect_info (dict): Perfect information from environment
            player_id (int): Current player ID
            game: Game object
            
        Returns:
            dict: Action probabilities from search
        '''
        # Get root node
        info_hash = self._hash_info_state(info_state)
        if info_hash not in self.search_trees[player_id]:
            root_node = MCTSNode(info_state, player_id)
            self.search_trees[player_id][info_hash] = root_node
        else:
            root_node = self.search_trees[player_id][info_hash]
        
        # Store initial trace length to restore state
        initial_trace_length = len(game.round.trace) if hasattr(game, 'round') and hasattr(game.round, 'trace') else 0
        
        # Enable step_back if not already enabled
        original_allow_step_back = getattr(game, 'allow_step_back', False)
        if not original_allow_step_back:
            game.allow_step_back = True
        
        # Perform simulations
        for _ in range(self.simulate_num):
            # Determinize
            deterministic_state = self._determinize(info_state, perfect_info)
            
            # Simulate
            self._simulate(game, player_id, deterministic_state, initial_trace_length)
            
            # Restore game state by stepping back
            # Count how many steps we need to go back
            current_trace_length = len(game.round.trace) if hasattr(game, 'round') and hasattr(game.round, 'trace') else 0
            steps_to_back = current_trace_length - initial_trace_length
            
            for _ in range(steps_to_back):
                if not game.step_back():
                    break
        
        # Restore original allow_step_back setting
        game.allow_step_back = original_allow_step_back
        
        # Return action probabilities based on visit counts
        legal_actions = info_state.get('actions', [])
        if not legal_actions:
            return {}
        
        total_visits = sum(root_node.N[action] for action in legal_actions)
        if total_visits == 0:
            # Fallback to uniform
            self._print_fallback_once("[Fallback] _get_action_probs_from_search: No visits in MCTS, using uniform distribution")
            uniform_prob = 1.0 / len(legal_actions)
            return {action: uniform_prob for action in legal_actions}
        
        probs = {}
        for action in legal_actions:
            probs[action] = root_node.N[action] / total_visits if total_visits > 0 else 0.0
        
        return probs
    
    def step(self, state):
        ''' Predict the action given the current state in generating training data.
        Uses FPMCTS to generate high-quality training data (state, policy, value).
        
        Args:
            state (dict): An dictionary that represents the current state
            
        Returns:
            action (int): The action predicted by FPMCTS
        '''
        # Try to get environment from state if available
        env = state.get('_env', None)
        
        # Get perfect information if env is available
        if env is not None:
            try:
                perfect_info = env.get_perfect_information()
            except:
                perfect_info = {}
        else:
            perfect_info = {}
        
        # Get current player ID and info state
        raw_obs = state.get('raw_obs', state)
        player_id = raw_obs.get('self', 0)
        info_state = state  # This should have 'obs' key from env
        
        # Get game object from env if available
        game = None
        if env is not None and hasattr(env, 'game'):
            game = env.game
            # Enable step_back for simulation
            if not game.allow_step_back:
                game.allow_step_back = True
        
        # If we don't have game access, fallback to network policy
        if game is None:
            # Fallback: use policy network directly
            self._print_fallback_once("[Fallback] step: No game access, using network policy directly")
            state_vector = self._extract_state_vector(state)
            legal_action_ids = list(state['legal_actions'].keys())
            legal_actions = state.get('raw_legal_actions', [])
            
            if not legal_actions:
                return np.random.choice(legal_action_ids)
            
            # Create legal actions mask
            legal_mask = np.zeros(self.num_actions)
            for action_str in legal_actions:
                if action_str in ACTION_2_ID:
                    legal_mask[ACTION_2_ID[action_str]] = 1.0
            
            # Get policy from network
            policy_probs, _ = self.policy_value_net.predict(state_vector, legal_mask)
            
            # Convert to action IDs and sample
            action_id_to_prob = {}
            for i, action_str in enumerate(legal_actions):
                if i < len(legal_action_ids) and action_str in ACTION_2_ID:
                    action_id = legal_action_ids[i]
                    action_id_to_prob[action_id] = policy_probs[ACTION_2_ID[action_str]]
            
            if action_id_to_prob:
                action_ids = list(action_id_to_prob.keys())
                probs = np.array([action_id_to_prob[aid] for aid in action_ids])
                if probs.sum() > 0:
                    probs = probs / probs.sum()
                    return np.random.choice(action_ids, p=probs)
            
            return np.random.choice(legal_action_ids)
        
        # Perform FPMCTS search
        try:
            action_probs = self._search(info_state, perfect_info, player_id, game)
        except Exception as e:
            # If search fails, fallback to network policy
            self._print_fallback_once(f"[Fallback] step: FPMCTS search failed: {e}, using network policy")
            state_vector = self._extract_state_vector(state)
            legal_action_ids = list(state['legal_actions'].keys())
            legal_actions = state.get('raw_legal_actions', [])
            
            if not legal_actions:
                return np.random.choice(legal_action_ids)
            
            legal_mask = np.zeros(self.num_actions)
            for action_str in legal_actions:
                if action_str in ACTION_2_ID:
                    legal_mask[ACTION_2_ID[action_str]] = 1.0
            
            policy_probs, _ = self.policy_value_net.predict(state_vector, legal_mask)
            action_id_to_prob = {}
            for i, action_str in enumerate(legal_actions):
                if i < len(legal_action_ids) and action_str in ACTION_2_ID:
                    action_id = legal_action_ids[i]
                    action_id_to_prob[action_id] = policy_probs[ACTION_2_ID[action_str]]
            
            if action_id_to_prob:
                action_ids = list(action_id_to_prob.keys())
                probs = np.array([action_id_to_prob[aid] for aid in action_ids])
                if probs.sum() > 0:
                    probs = probs / probs.sum()
                    return np.random.choice(action_ids, p=probs)
            
            return np.random.choice(legal_action_ids)
        
        # Select action based on FPMCTS probabilities
        legal_action_ids = list(state['legal_actions'].keys())
        legal_actions = state.get('raw_legal_actions', [])
        
        if not action_probs:
            # Fallback to random
            self._print_fallback_once("[Fallback] step: No action probabilities from search, using random action")
            return np.random.choice(legal_action_ids)
        
        # Convert action strings to IDs and select based on probabilities
        action_id_to_prob = {}
        for i, action_str in enumerate(legal_actions):
            if i < len(legal_action_ids) and action_str in action_probs:
                action_id_to_prob[legal_action_ids[i]] = action_probs[action_str]
        
        if action_id_to_prob:
            # Sample based on probabilities
            action_ids = list(action_id_to_prob.keys())
            probs = list(action_id_to_prob.values())
            probs = np.array(probs)
            if probs.sum() > 0:
                probs = probs / probs.sum()  # Normalize
                return np.random.choice(action_ids, p=probs)
        
        # Fallback to random
        self._print_fallback_once("[Fallback] step: Failed to convert action probabilities, using random action")
        return np.random.choice(legal_action_ids)
    
    def eval_step(self, state):
        ''' Predict the action given the current state for evaluation using FPMCTS
        
        Args:
            state (dict): An dictionary that represents the current state
            
        Returns:
            action (int): The action predicted by FPMCTS
            probs (dict): The dictionary of action probabilities
        '''
        # Try to get environment from state if available
        # Some environments may store a reference to the env in state
        env = state.get('_env', None)
        
        # Get perfect information if env is available
        if env is not None:
            try:
                perfect_info = env.get_perfect_information()
            except Exception as e:
                self._print_fallback_once(f"[Fallback] eval_step: Failed to get perfect information: {e}, using empty dict")
                perfect_info = {}
        else:
            perfect_info = {}
        
        # Get current player ID and info state
        # State from env has 'raw_obs' which contains the game state
        raw_obs = state.get('raw_obs', state)
        player_id = raw_obs.get('self', 0)
        
        # For network input, we need the extracted state with 'obs'
        # Use the state dict directly which should have 'obs' from env._extract_state
        info_state = state  # This should have 'obs' key from env
        
        # Get game object from env if available
        game = None
        if env is not None and hasattr(env, 'game'):
            game = env.game
            # Enable step_back for simulation
            if not game.allow_step_back:
                game.allow_step_back = True
        
        # If we don't have game access, use a simple fallback strategy
        if game is None:
            # Fallback: use uniform random or simple heuristic
            self._print_fallback_once("[Fallback] eval_step: No game access, using random action")
            legal_action_ids = list(state['legal_actions'].keys())
            action_id = np.random.choice(legal_action_ids)
            
            # Build uniform probabilities
            legal_actions = state.get('raw_legal_actions', [])
            uniform_prob = 1.0 / len(legal_actions) if legal_actions else 0.0
            info = {
                'probs': {action_str: uniform_prob for action_str in legal_actions}
            }
            return action_id, info
        
        # Perform FPMCTS search
        try:
            action_probs = self._search(info_state, perfect_info, player_id, game)
        except Exception as e:
            # If search fails, fallback to random
            self._print_fallback_once(f"[Fallback] eval_step: FPMCTS search failed: {e}, using random action")
            legal_action_ids = list(state['legal_actions'].keys())
            action_id = np.random.choice(legal_action_ids)
            legal_actions = state.get('raw_legal_actions', [])
            uniform_prob = 1.0 / len(legal_actions) if legal_actions else 0.0
            info = {
                'probs': {action_str: uniform_prob for action_str in legal_actions}
            }
            return action_id, info
        
        # Select action based on probabilities
        legal_action_ids = list(state['legal_actions'].keys())
        legal_actions = state.get('raw_legal_actions', [])
        
        if not action_probs:
            # Fallback to random
            self._print_fallback_once("[Fallback] eval_step: No action probabilities from search, using random action")
            action_id = np.random.choice(legal_action_ids)
        else:
            # Convert action strings to IDs and select
            action_id_to_prob = {}
            for i, action_str in enumerate(legal_actions):
                if i < len(legal_action_ids) and action_str in action_probs:
                    action_id_to_prob[legal_action_ids[i]] = action_probs[action_str]
            
            if action_id_to_prob:
                # Sample based on probabilities
                action_ids = list(action_id_to_prob.keys())
                probs = list(action_id_to_prob.values())
                probs = np.array(probs)
                if probs.sum() > 0:
                    probs = probs / probs.sum()  # Normalize
                    action_id = np.random.choice(action_ids, p=probs)
                else:
                    action_id = np.random.choice(action_ids)
            else:
                action_id = np.random.choice(legal_action_ids)
        
        # Build info dict
        info = {}
        info['probs'] = {action_str: action_probs.get(action_str, 0.0) 
                         for action_str in legal_actions}
        
        return action_id, info
    
    def load_model(self, model_path):
        ''' Load pretrained model
        
        Args:
            model_path (str): Path to model file
        '''
        import torch
        checkpoint = torch.load(model_path, map_location=self.policy_value_net.device)
        self.policy_value_net.load_state_dict(checkpoint['model_state_dict'])
        self.policy_value_net.eval()
        print(f"Loaded model from {model_path}")
    
    def save_model(self, model_path):
        ''' Save model
        
        Args:
            model_path (str): Path to save model
        '''
        import torch
        checkpoint = {
            'model_state_dict': self.policy_value_net.state_dict(),
            'state_dim': self.state_dim,
            'num_actions': self.num_actions
        }
        torch.save(checkpoint, model_path)
        print(f"Saved model to {model_path}")
    
    def init_trainer(self, learning_rate=0.001, value_loss_coef=1.0, l2_reg_coef=1e-4):
        ''' Initialize trainer for the network
        
        Args:
            learning_rate (float): Learning rate
            value_loss_coef (float): Value loss coefficient
            l2_reg_coef (float): L2 regularization coefficient
        '''
        # Import here to avoid circular import with trainer.py
        from rlcard.agents.deltadou.trainer import PolicyValueTrainer
        
        self.trainer = PolicyValueTrainer(
            self.policy_value_net,
            learning_rate=learning_rate,
            value_loss_coef=value_loss_coef,
            l2_reg_coef=l2_reg_coef,
            device=self.policy_value_net.device
        )
    
    def train_network(self, batch_data):
        ''' Train the policy-value network on a batch of data
        
        Args:
            batch_data (list): List of (state, target_policy, target_value) tuples
            
        Returns:
            loss (float): Training loss
        '''
        if self.trainer is None:
            self.init_trainer()
        
        return self.trainer.train_batch(batch_data)
