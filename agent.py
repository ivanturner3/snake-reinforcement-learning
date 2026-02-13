"""
Agent Module for Snake Game Reinforcement Learning

This module implements a Deep Q-Learning agent that learns to play the Snake game.
The agent uses a neural network to approximate Q-values and employs experience replay
for stable learning.
"""

import torch
import random
import numpy as np
from collections import deque
from game import SnakeGameAI, Direction, Point
from model import Linear_QNet, QTrainer
from helper import plot

# Hyperparameters for training
MAX_MEMORY = 100_000  # Maximum number of experiences stored in replay memory
BATCH_SIZE = 1000     # Number of experiences sampled for training each iteration
LR = 0.001            # Learning rate for the neural network optimizer

class Agent:
    """
    Deep Q-Learning Agent for Snake Game
    
    This agent uses a neural network to learn optimal actions in the Snake game.
    It implements epsilon-greedy exploration and experience replay for training.
    """

    def __init__(self):
        """
        Initialize the agent with model parameters and training settings.
        """
        self.n_games = 0                      # Counter for number of games played
        self.epsilon = 0                      # Controls exploration vs exploitation (randomness)
        self.gamma = 0.9                      # Discount rate for future rewards (0-1)
        self.memory = deque(maxlen=MAX_MEMORY) # Experience replay memory with automatic removal of old experiences
        
        # Neural network: 11 input features -> 256 hidden neurons -> 3 output actions
        self.model = Linear_QNet(11, 256, 3)
        
        # Q-learning trainer that handles the neural network optimization
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)


    def get_state(self, game):
        """
        Extract the current state of the game as an 11-element feature vector.
        
        The state consists of:
        - Danger detection (3 features): straight, right, left
        - Current direction (4 features): left, right, up, down (one-hot encoded)
        - Food location (4 features): left, right, up, down relative to snake head
        
        Args:
            game: SnakeGameAI instance containing current game state
            
        Returns:
            numpy.ndarray: 11-element state vector with binary/boolean values
        """
        head = game.snake[0]  # Get the current position of the snake's head
        
        # Define points adjacent to the head in all four directions (20 pixels = 1 block)
        point_l = Point(head.x - 20, head.y)  # Left point
        point_r = Point(head.x + 20, head.y)  # Right point
        point_u = Point(head.x, head.y - 20)  # Up point
        point_d = Point(head.x, head.y + 20)  # Down point
        
        # Determine current direction of movement (boolean flags)
        dir_l = game.direction == Direction.LEFT
        dir_r = game.direction == Direction.RIGHT
        dir_u = game.direction == Direction.UP
        dir_d = game.direction == Direction.DOWN

        # Build the 11-element state vector
        state = [
            # Danger straight: Is there a collision danger in the current direction?
            (dir_r and game.is_collision(point_r)) or 
            (dir_l and game.is_collision(point_l)) or 
            (dir_u and game.is_collision(point_u)) or 
            (dir_d and game.is_collision(point_d)),

            # Danger right: Is there danger to the right of current direction?
            (dir_u and game.is_collision(point_r)) or  # If moving up, right is to the right
            (dir_d and game.is_collision(point_l)) or  # If moving down, left is to the right
            (dir_l and game.is_collision(point_u)) or  # If moving left, up is to the right
            (dir_r and game.is_collision(point_d)),    # If moving right, down is to the right

            # Danger left: Is there danger to the left of current direction?
            (dir_d and game.is_collision(point_r)) or  # If moving down, right is to the left
            (dir_u and game.is_collision(point_l)) or  # If moving up, left is to the left
            (dir_r and game.is_collision(point_u)) or  # If moving right, up is to the left
            (dir_l and game.is_collision(point_d)),    # If moving left, down is to the left
            
            # Move direction (one-hot encoded current direction)
            dir_l,  # Currently moving left
            dir_r,  # Currently moving right
            dir_u,  # Currently moving up
            dir_d,  # Currently moving down
            
            # Food location relative to head
            game.food.x < game.head.x,  # Food is to the left
            game.food.x > game.head.x,  # Food is to the right
            game.food.y < game.head.y,  # Food is above (y decreases upward)
            game.food.y > game.head.y   # Food is below (y increases downward)
            ]

        return np.array(state, dtype=int)

    def remember(self, state, action, reward, next_state, done):
        """
        Store an experience in the replay memory.
        
        Experience replay is a key technique in Deep Q-Learning that breaks the
        correlation between consecutive training samples and improves stability.
        
        Args:
            state: Current state observation
            action: Action taken in the current state
            reward: Reward received after taking the action
            next_state: Resulting state after the action
            done: Boolean indicating if the episode ended
        """
        # Append experience tuple to memory; oldest experiences are automatically
        # removed when MAX_MEMORY is reached (deque with maxlen)
        self.memory.append((state, action, reward, next_state, done))

    def train_long_memory(self):
        """
        Train the neural network using a batch of experiences from replay memory.
        
        This method is called after each game ends. It samples a random batch of
        experiences from memory and performs a training step. Using random samples
        breaks temporal correlations in the data, leading to more stable learning.
        """
        if len(self.memory) > BATCH_SIZE:
            # Sample a random batch of experiences for training
            mini_sample = random.sample(self.memory, BATCH_SIZE)
        else:
            # Use all available experiences if we don't have enough for a full batch
            mini_sample = self.memory

        # Unzip the batch of experiences into separate arrays
        states, actions, rewards, next_states, dones = zip(*mini_sample)
        
        # Perform a training step with the batch
        self.trainer.train_step(states, actions, rewards, next_states, dones)

    def train_short_memory(self, state, action, reward, next_state, done):
        """
        Train the neural network on a single experience immediately.
        
        Short memory training (also called online learning) updates the model
        after each step, allowing the agent to quickly adapt to recent experiences.
        
        Args:
            state: Current state observation
            action: Action taken
            reward: Reward received
            next_state: Resulting state
            done: Whether the episode ended
        """
        self.trainer.train_step(state, action, reward, next_state, done)

    def get_action(self, state):
        """
        Decide which action to take based on the current state.
        
        Uses epsilon-greedy strategy:
        - With probability epsilon: take a random action (exploration)
        - Otherwise: take the action with highest predicted Q-value (exploitation)
        
        Epsilon decreases as more games are played, shifting from exploration
        to exploitation over time.
        
        Args:
            state: Current game state as a feature vector
            
        Returns:
            list: One-hot encoded action [straight, right turn, left turn]
        """
        # Epsilon decreases linearly with number of games played
        # Early games: more exploration (random moves)
        # Later games: more exploitation (using learned policy)
        self.epsilon = 80 - self.n_games
        
        # Initialize action as [straight, right, left] - all zeros
        final_move = [0, 0, 0]
        
        if random.randint(0, 200) < self.epsilon:
            # Exploration: take a random action
            move = random.randint(0, 2)
            final_move[move] = 1
        else:
            # Exploitation: use the neural network to predict best action
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)  # Get Q-values for all actions
            move = torch.argmax(prediction).item()  # Choose action with highest Q-value
            final_move[move] = 1

        return final_move


def train():
    """
    Main training loop for the Deep Q-Learning agent.
    
    This function implements the complete training algorithm:
    1. Initialize agent and game environment
    2. For each step:
       - Get current state
       - Choose action (epsilon-greedy)
       - Execute action and observe reward
       - Store experience in memory
       - Train on recent experience (short memory)
    3. After each game ends:
       - Train on batch from replay memory (long memory)
       - Save model if new record is achieved
       - Plot progress
    
    The training continues indefinitely until manually stopped.
    """
    # Lists to track performance over time
    plot_scores = []       # Score for each game
    plot_mean_scores = []  # Running average score
    total_score = 0        # Sum of all scores
    record = 0             # Best score achieved
    
    # Initialize the agent and game environment
    agent = Agent()
    game = SnakeGameAI()
    
    # Main training loop (runs indefinitely)
    while True:
        # Get current state of the game
        state_old = agent.get_state(game)

        # Decide on an action based on current state
        final_move = agent.get_action(state_old)

        # Perform the action and get the result
        reward, done, score = game.play_step(final_move)
        state_new = agent.get_state(game)

        # Train on this single experience (short-term memory / online learning)
        agent.train_short_memory(state_old, final_move, reward, state_new, done)

        # Store the experience in replay memory for batch training later
        agent.remember(state_old, final_move, reward, state_new, done)

        if done:
            # Game over - perform end-of-game training and bookkeeping
            
            # Reset the game for a new round
            game.reset()
            agent.n_games += 1
            
            # Train on a batch of experiences from replay memory (long-term memory)
            agent.train_long_memory()

            # Save the model if we achieved a new high score
            if score > record:
                record = score
                agent.model.save()

            # Log progress to console
            print('Game', agent.n_games, 'Score', score, 'Record:', record)

            # Update performance tracking
            plot_scores.append(score)
            total_score += score
            mean_score = total_score / agent.n_games
            plot_mean_scores.append(mean_score)
            
            # Visualize training progress
            plot(plot_scores, plot_mean_scores)


if __name__ == '__main__':
    train()