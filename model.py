"""
Neural Network Model for Deep Q-Learning

This module implements the Q-Network and training components for the
reinforcement learning agent. It uses PyTorch to build a simple feedforward
neural network that approximates Q-values.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os

class Linear_QNet(nn.Module):
    """
    Q-Network: A feedforward neural network for Q-value approximation.
    
    This network takes the game state as input and outputs Q-values for each
    possible action. The agent selects actions based on these Q-values.
    
    Architecture:
        Input Layer: state features (e.g., 11 features)
        Hidden Layer: ReLU activated neurons (e.g., 256 neurons)
        Output Layer: Q-values for each action (e.g., 3 actions)
    """
    
    def __init__(self, input_size, hidden_size, output_size):
        """
        Initialize the neural network layers.
        
        Args:
            input_size: Number of input features (state dimensions)
            hidden_size: Number of neurons in the hidden layer
            output_size: Number of possible actions
        """
        super().__init__()
        # First layer: input -> hidden
        self.linear1 = nn.Linear(input_size, hidden_size)
        # Second layer: hidden -> output
        self.linear2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor (state)
            
        Returns:
            Tensor of Q-values for each action
        """
        # Apply ReLU activation after first layer
        x = F.relu(self.linear1(x))
        # Output layer (no activation - raw Q-values)
        x = self.linear2(x)
        return x

    def save(self, file_name='model.pth'):
        """
        Save the model's learned parameters to disk.
        
        Saves the state_dict containing all network weights and biases.
        Creates the model directory if it doesn't exist.
        
        Args:
            file_name: Name of the file to save (default: 'model.pth')
        """
        model_folder_path = './model'
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)

        file_name = os.path.join(model_folder_path, file_name)
        torch.save(self.state_dict(), file_name)


class QTrainer:
    """
    Q-Learning Trainer
    
    This class handles the training process for the Q-Network using the
    Bellman equation. It implements the Deep Q-Learning algorithm with
    experience replay.
    
    The Bellman equation used:
        Q(s, a) = r + γ * max(Q(s', a'))
    where:
        - Q(s, a) is the Q-value for state s and action a
        - r is the immediate reward
        - γ (gamma) is the discount factor
        - s' is the next state
        - a' are possible actions in the next state
    """
    
    def __init__(self, model, lr, gamma):
        """
        Initialize the trainer with a model and hyperparameters.
        
        Args:
            model: The Q-Network to train
            lr: Learning rate for the optimizer
            gamma: Discount factor for future rewards (0 to 1)
        """
        self.lr = lr
        self.gamma = gamma
        self.model = model
        # Adam optimizer for updating network weights
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr)
        # Mean Squared Error loss for Q-value prediction
        self.criterion = nn.MSELoss()

    def train_step(self, state, action, reward, next_state, done):
        """
        Perform one training step using the Q-learning algorithm.
        
        This method:
        1. Converts inputs to tensors
        2. Predicts Q-values for current states
        3. Calculates target Q-values using Bellman equation
        4. Computes loss between predicted and target Q-values
        5. Updates network weights via backpropagation
        
        Args:
            state: Current state(s) - can be single state or batch
            action: Action(s) taken - one-hot encoded
            reward: Reward(s) received
            next_state: Resulting state(s)
            done: Boolean(s) indicating if episode ended
        """
        # Convert inputs to PyTorch tensors
        state = torch.tensor(state, dtype=torch.float)
        next_state = torch.tensor(next_state, dtype=torch.float)
        action = torch.tensor(action, dtype=torch.long)
        reward = torch.tensor(reward, dtype=torch.float)

        # Handle both single samples and batches
        if len(state.shape) == 1:
            # Single sample - reshape to batch of size 1
            state = torch.unsqueeze(state, 0)
            next_state = torch.unsqueeze(next_state, 0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            done = (done, )  # Convert to tuple for consistent iteration

        # Step 1: Get predicted Q-values for current states
        pred = self.model(state)

        # Step 2: Calculate target Q-values using Bellman equation
        target = pred.clone()
        for idx in range(len(done)):
            Q_new = reward[idx]
            if not done[idx]:
                # If episode not done, add discounted future reward
                # Q_new = reward + gamma * max(next Q-values)
                Q_new = reward[idx] + self.gamma * torch.max(self.model(next_state[idx]))

            # Update the Q-value for the action that was taken
            target[idx][torch.argmax(action[idx]).item()] = Q_new
    
        # Step 3: Calculate loss between predicted and target Q-values
        # Step 4: Perform backpropagation and update weights
        self.optimizer.zero_grad()  # Clear previous gradients
        loss = self.criterion(target, pred)  # Calculate MSE loss
        loss.backward()  # Compute gradients

        self.optimizer.step()  # Update network weights