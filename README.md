# Snake Reinforcement Learning 🐍🤖

A Deep Q-Learning implementation that trains an AI agent to master the classic Snake game. Watch as the AI learns from scratch, gradually improving its strategy through trial and error using neural networks and reinforcement learning.

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [How It Works](#how-it-works)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [The Learning Process](#the-learning-process)
- [Model Architecture](#model-architecture)
- [Hyperparameters](#hyperparameters)
- [Results](#results)
- [Requirements](#requirements)
- [Contributing](#contributing)

## 🎯 Overview

This project implements a **Deep Q-Network (DQN)** agent that learns to play Snake through reinforcement learning. The agent starts with no knowledge of the game and gradually learns optimal strategies by playing thousands of games, receiving rewards for eating food and penalties for colliding with walls or itself.

Unlike traditional programming where we explicitly code the game strategy, this AI learns the strategy on its own by:
- **Observing** the game state
- **Taking actions** (straight, turn right, turn left)
- **Receiving rewards** (+10 for food, -10 for collision)
- **Learning** from experience to maximize future rewards

## ✨ Features

- **Deep Q-Learning**: Implements the DQN algorithm with experience replay
- **Neural Network**: PyTorch-based Q-Network for value approximation
- **Real-time Visualization**: Live plotting of training progress
- **Human Playable Version**: Play the game yourself to compare with the AI
- **Model Persistence**: Automatically saves the best-performing model
- **Comprehensive Documentation**: Detailed comments throughout the codebase

## 🧠 How It Works

### The Reinforcement Learning Cycle

1. **State Observation** (11 features):
   - Danger detection in 3 directions (straight, right, left)
   - Current movement direction (4 values, one-hot encoded)
   - Food location relative to snake head (4 values)

2. **Action Selection**: 
   - **Exploration**: Random actions early in training (high epsilon)
   - **Exploitation**: Use neural network predictions later (low epsilon)
   - Actions: [straight, turn right, turn left]

3. **Reward System**:
   - +10 points for eating food
   - -10 points for collision (death)
   - 0 points for normal moves

4. **Learning**:
   - **Short-term memory**: Train on each move immediately
   - **Long-term memory**: Train on batches from replay memory after each game
   - Uses Bellman equation: Q(s,a) = r + γ * max(Q(s',a'))

### Deep Q-Learning Components

- **Q-Network**: Approximates Q-values (expected future rewards) for each action
- **Experience Replay**: Stores past experiences and samples random batches for training
- **Epsilon-Greedy Strategy**: Balances exploration (trying new things) vs exploitation (using learned knowledge)
- **Target Q-Values**: Uses Bellman equation to compute optimal Q-values

## 📁 Project Structure

```
snake-reinforcement-learning/
│
├── agent.py              # DQN agent implementation
│   ├── Agent class       # Main RL agent
│   ├── get_state()       # Extract game state features
│   ├── get_action()      # Epsilon-greedy action selection
│   ├── remember()        # Store experiences in memory
│   ├── train_short_memory()  # Train on single experience
│   ├── train_long_memory()   # Train on batch from replay memory
│   └── train()           # Main training loop
│
├── game.py               # Snake game environment for AI
│   ├── SnakeGameAI class # Game logic and rendering
│   ├── play_step()       # Execute one game step
│   ├── is_collision()    # Collision detection
│   └── reset()           # Reset game state
│
├── model.py              # Neural network and trainer
│   ├── Linear_QNet       # Q-Network architecture
│   ├── forward()         # Neural network forward pass
│   ├── save()            # Save model weights
│   └── QTrainer          # Training logic with Bellman equation
│
├── helper.py             # Visualization utilities
│   └── plot()            # Real-time training progress plotting
│
├── snake_game_human.py   # Human-playable version
│   └── SnakeGame class   # Keyboard-controlled Snake game
│
├── model/
│   └── model.pth         # Saved neural network weights
│
└── README.md             # This file
```

## 🔧 Installation

### Prerequisites

- Python 3.7 or higher
- pip package manager

### Install Dependencies

```bash
pip install torch pygame matplotlib ipython numpy
```

Or create a `requirements.txt`:

```txt
torch>=1.9.0
pygame>=2.0.0
matplotlib>=3.4.0
ipython>=7.25.0
numpy>=1.21.0
```

Then install:

```bash
pip install -r requirements.txt
```

## 🚀 Usage

### Train the AI Agent

Run the main training script to start teaching the AI:

```bash
python agent.py
```

**What to expect**:
- A game window will open showing the snake playing
- A matplotlib plot window will show training progress
- Console output will display: Game number, Score, Record score
- The model will save automatically when beating the previous record
- Training runs indefinitely - stop with Ctrl+C when satisfied

**Training Tips**:
- Let it run for at least 100-200 games to see significant learning
- Early games: Random movements (exploration)
- Later games: Increasingly strategic movements (exploitation)
- The plot shows individual scores (blue) and running average (orange)

### Play the Game Yourself

Test your skills against the AI:

```bash
python snake_game_human.py
```

**Controls**:
- ⬆️ Arrow Up: Move up
- ⬇️ Arrow Down: Move down
- ⬅️ Arrow Left: Move left
- ➡️ Arrow Right: Move right

### Load a Trained Model

To continue training from a saved model, modify [agent.py](agent.py#L21-L22):

```python
self.model = Linear_QNet(11, 256, 3)
self.model.load_state_dict(torch.load('./model/model.pth'))
```

## 📚 The Learning Process

### Phase 1: Random Exploration (Games 1-80)

- **Epsilon**: High (starts at 80)
- **Behavior**: Mostly random moves
- **Purpose**: Explore the environment and gather diverse experiences
- **Typical Score**: 0-5

### Phase 2: Learning Transition (Games 80-150)

- **Epsilon**: Decreasing
- **Behavior**: Mix of random and learned moves
- **Purpose**: Balance exploration with exploitation
- **Typical Score**: 5-15

### Phase 3: Exploitation (Games 150+)

- **Epsilon**: Low (approaches 0)
- **Behavior**: Mostly uses learned strategy
- **Purpose**: Refine and optimize learned policy
- **Typical Score**: 15-40+

### Why the AI Improves

1. **Experience Replay**: Learns from random samples of past experiences, breaking correlations
2. **Bellman Equation**: Learns to predict long-term rewards, not just immediate ones
3. **Neural Network**: Generalizes from specific states to similar situations
4. **Epsilon Decay**: Shifts from exploration to exploitation as it learns

## 🏗️ Model Architecture

### Q-Network Structure

```
Input Layer:    11 neurons (state features)
                ↓
Hidden Layer:   256 neurons (ReLU activation)
                ↓
Output Layer:   3 neurons (Q-values for each action)
```

### State Representation (11 features)

| Feature Index | Description |
|--------------|-------------|
| 0 | Danger straight ahead |
| 1 | Danger to the right |
| 2 | Danger to the left |
| 3 | Moving left |
| 4 | Moving right |
| 5 | Moving up |
| 6 | Moving down |
| 7 | Food is to the left |
| 8 | Food is to the right |
| 9 | Food is above |
| 10 | Food is below |

### Action Space (one-hot encoded)

| Action | Encoding |
|--------|----------|
| Straight | [1, 0, 0] |
| Turn Right | [0, 1, 0] |
| Turn Left | [0, 0, 1] |

## ⚙️ Hyperparameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| `MAX_MEMORY` | 100,000 | Maximum experiences stored in replay memory |
| `BATCH_SIZE` | 1,000 | Number of experiences sampled per training iteration |
| `LR` | 0.001 | Learning rate for Adam optimizer |
| `GAMMA` | 0.9 | Discount factor for future rewards (0-1) |
| `EPSILON` | 80 - n_games | Exploration rate (decreases linearly) |
| `HIDDEN_SIZE` | 256 | Number of neurons in hidden layer |
| `SPEED` | 800 FPS | Game speed during training |

### Tuning Guide

- **Increase `GAMMA`** (→0.95): Agent values future rewards more (better long-term planning)
- **Increase `BATCH_SIZE`**: More stable learning but slower training
- **Increase `HIDDEN_SIZE`**: More capacity to learn complex patterns (risk of overfitting)
- **Decrease `LR`**: More stable but slower learning
- **Adjust epsilon decay**: Change `80` to explore more/less games

## 📊 Results

### Typical Training Progress

- **Games 0-50**: Score mostly 0-3 (random exploration)
- **Games 50-100**: Score 3-10 (beginning to learn patterns)
- **Games 100-200**: Score 10-20 (learning to pursue food)
- **Games 200+**: Score 20-50+ (optimizing survival and food collection)

### Performance Metrics

The AI typically achieves:
- **Average Score**: 20-30 after 200 games
- **Record Score**: 40-80 (varies by training duration)
- **Success Rate**: >80% of games score above 10 (after 300 games)

## 📦 Requirements

```
Python 3.7+
torch>=1.9.0          # PyTorch for neural networks
pygame>=2.0.0         # Game rendering
matplotlib>=3.4.0     # Plotting training progress
ipython>=7.25.0       # Interactive plotting support
numpy>=1.21.0         # Numerical operations
```

## 🤝 Contributing

Contributions are welcome! Here are some ideas:

### Potential Improvements

- [ ] Implement Double DQN to reduce overestimation bias
- [ ] Add dueling network architecture
- [ ] Implement prioritized experience replay
- [ ] Add more sophisticated state representation
- [ ] Create web-based visualization
- [ ] Add tensorboard logging
- [ ] Implement curriculum learning

### How to Contribute

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/improvement`)
3. Commit your changes (`git commit -am 'Add new feature'`)
4. Push to the branch (`git push origin feature/improvement`)
5. Create a Pull Request

## 📖 Further Reading

### Deep Q-Learning Resources

- [Playing Atari with Deep Reinforcement Learning](https://arxiv.org/abs/1312.5602) - Original DQN paper
- [Human-level control through deep reinforcement learning](https://www.nature.com/articles/nature14236) - Nature DQN paper
- [Deep Reinforcement Learning Course](https://huggingface.co/deep-rl-course/unit0/introduction) - Comprehensive RL course

### Related Concepts

- **Markov Decision Process (MDP)**: Mathematical framework for RL
- **Bellman Equation**: Foundation of Q-learning
- **Temporal Difference Learning**: Learning from differences in predictions
- **Experience Replay**: Key technique for stable DQN training

## 📝 License

This project is open source and available for educational purposes.

## 🙏 Acknowledgments

- Inspired by classic reinforcement learning papers and tutorials
- Built with PyTorch and Pygame
- Thanks to the RL community for excellent resources

---

**Happy Learning! 🐍🤖**

*Watch your AI evolve from a confused beginner to a Snake master!*
