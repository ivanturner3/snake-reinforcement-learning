"""
Snake Game AI Environment

This module implements the Snake game environment for AI training.
It provides a gym-like interface for the agent to interact with the game,
using Pygame for visualization.
"""

import pygame
import random
from enum import Enum
from collections import namedtuple
import numpy as np

# Initialize Pygame
pygame.init()
font = pygame.font.SysFont('arial', 25)

class Direction(Enum):
    """Enumeration for the four possible movement directions."""
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4

# Named tuple for representing 2D coordinates
Point = namedtuple('Point', 'x, y')

# RGB color constants for game rendering
WHITE = (255, 255, 255)
RED = (200, 0, 0)      # Food color
BLUE1 = (0, 0, 255)    # Snake body primary color
BLUE2 = (0, 100, 255)  # Snake body secondary color (for gradient effect)
BLACK = (0, 0, 0)      # Background color

# Game configuration constants
BLOCK_SIZE = 20  # Size of each grid cell in pixels
SPEED = 800      # Game speed (FPS) - high value for fast AI training

class SnakeGameAI:
    """
    Snake Game Environment for AI Training
    
    This class implements the Snake game with an interface designed for
    reinforcement learning agents. It handles game logic, rendering, and
    provides structured state/reward information for training.
    """

    def __init__(self, w=640, h=480):
        """
        Initialize the game window and start a new game.
        
        Args:
            w: Window width in pixels (default: 640)
            h: Window height in pixels (default: 480)
        """
        self.w = w
        self.h = h
        
        # Initialize Pygame display window
        self.display = pygame.display.set_mode((self.w, self.h))
        pygame.display.set_caption('Snake')
        self.clock = pygame.time.Clock()
        
        # Initialize the game state
        self.reset()


    def reset(self):
        """
        Reset the game to its initial state.
        
        Called at the start of each new game. Initializes the snake at the
        center of the screen, places food randomly, and resets all counters.
        """
        # Set initial movement direction to right
        self.direction = Direction.RIGHT

        # Initialize snake at center of screen with 3 segments
        self.head = Point(self.w/2, self.h/2)
        self.snake = [
            self.head,                                      # Head
            Point(self.head.x - BLOCK_SIZE, self.head.y),   # Body segment 1
            Point(self.head.x - (2*BLOCK_SIZE), self.head.y) # Body segment 2
        ]

        # Reset score and counters
        self.score = 0
        self.food = None
        self._place_food()        # Place first food item
        self.frame_iteration = 0  # Counter to detect if snake is stuck


    def _place_food(self):
        """
        Place food at a random location on the grid.
        
        Ensures food is aligned to the grid (multiples of BLOCK_SIZE) and
        doesn't spawn on the snake's body. Uses recursion to retry if food
        spawns on the snake.
        """
        # Generate random coordinates aligned to the grid
        x = random.randint(0, (self.w - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
        y = random.randint(0, (self.h - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
        self.food = Point(x, y)
        
        # If food spawned on the snake, try again recursively
        if self.food in self.snake:
            self._place_food()


    def play_step(self, action):
        """
        Execute one step of the game based on the agent's action.
        
        This is the main interface between the game and the AI agent. It:
        1. Processes pygame events
        2. Moves the snake based on the action
        3. Checks for collisions
        4. Handles food consumption
        5. Updates the display
        6. Returns reward, game_over status, and score
        
        Args:
            action: One-hot encoded action [straight, right_turn, left_turn]
            
        Returns:
            tuple: (reward, game_over, score)
                - reward (int): Reward for this step (+10 for food, -10 for collision, 0 otherwise)
                - game_over (bool): True if the game ended
                - score (int): Current score
        """
        self.frame_iteration += 1
        
        # 1. Collect user input (handle window close events)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
        
        # 2. Move the snake based on the action
        self._move(action)  # Update head position
        self.snake.insert(0, self.head)  # Add new head to snake
        
        # 3. Check if game over
        reward = 0
        game_over = False
        
        # Game ends if snake collides with wall/itself OR gets stuck in a loop
        # The frame_iteration check prevents the snake from wandering indefinitely
        if self.is_collision() or self.frame_iteration > 100 * len(self.snake):
            game_over = True
            reward = -10  # Negative reward for dying
            return reward, game_over, self.score

        # 4. Check if snake ate food or just moved
        if self.head == self.food:
            # Snake ate food - increase score and spawn new food
            self.score += 1
            reward = 10  # Positive reward for eating food
            self._place_food()
        else:
            # Normal move - remove tail segment (snake doesn't grow)
            self.snake.pop()
        
        # 5. Update the display and control game speed
        self._update_ui()
        self.clock.tick(SPEED)
        
        # 6. Return game state information
        return reward, game_over, self.score


    def is_collision(self, pt=None):
        """
        Check if a point collides with walls or the snake's body.
        
        Used both to check current collisions and to predict future collisions
        for state representation.
        
        Args:
            pt: Point to check for collision. If None, checks the snake's head.
            
        Returns:
            bool: True if collision detected, False otherwise
        """
        if pt is None:
            pt = self.head
            
        # Check collision with boundaries
        if pt.x > self.w - BLOCK_SIZE or pt.x < 0 or pt.y > self.h - BLOCK_SIZE or pt.y < 0:
            return True
            
        # Check collision with snake's own body (excluding head)
        if pt in self.snake[1:]:
            return True

        return False


    def _update_ui(self):
        """
        Render the current game state to the display.
        
        Draws the background, snake (with gradient effect), food, and score.
        """
        # Fill background with black
        self.display.fill(BLACK)

        # Draw each segment of the snake with a two-tone effect
        for pt in self.snake:
            # Outer rectangle (darker blue)
            pygame.draw.rect(self.display, BLUE1, pygame.Rect(pt.x, pt.y, BLOCK_SIZE, BLOCK_SIZE))
            # Inner rectangle (lighter blue) - creates depth effect
            pygame.draw.rect(self.display, BLUE2, pygame.Rect(pt.x+4, pt.y+4, 12, 12))

        # Draw food as a red rectangle
        pygame.draw.rect(self.display, RED, pygame.Rect(self.food.x, self.food.y, BLOCK_SIZE, BLOCK_SIZE))

        # Render and display the score
        text = font.render("Score: " + str(self.score), True, WHITE)
        self.display.blit(text, [0, 0])
        
        # Update the display
        pygame.display.flip()


    def _move(self, action):
        """
        Update the snake's direction and head position based on the action.
        
        Actions are relative to current direction:
        - [1, 0, 0]: Continue straight
        - [0, 1, 0]: Turn right (clockwise)
        - [0, 0, 1]: Turn left (counter-clockwise)
        
        Args:
            action: One-hot encoded action array
        """
        # Define clockwise direction order
        clock_wise = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
        idx = clock_wise.index(self.direction)

        # Determine new direction based on action
        if np.array_equal(action, [1, 0, 0]):
            # Go straight - no direction change
            new_dir = clock_wise[idx]
        elif np.array_equal(action, [0, 1, 0]):
            # Turn right - next direction clockwise
            next_idx = (idx + 1) % 4
            new_dir = clock_wise[next_idx]  # RIGHT -> DOWN -> LEFT -> UP -> RIGHT
        else:  # [0, 0, 1]
            # Turn left - next direction counter-clockwise
            next_idx = (idx - 1) % 4
            new_dir = clock_wise[next_idx]  # RIGHT -> UP -> LEFT -> DOWN -> RIGHT

        self.direction = new_dir

        # Calculate new head position based on direction
        x = self.head.x
        y = self.head.y
        if self.direction == Direction.RIGHT:
            x += BLOCK_SIZE
        elif self.direction == Direction.LEFT:
            x -= BLOCK_SIZE
        elif self.direction == Direction.DOWN:
            y += BLOCK_SIZE
        elif self.direction == Direction.UP:
            y -= BLOCK_SIZE

        self.head = Point(x, y)