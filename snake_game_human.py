"""
Snake Game - Human Playable Version

This is a classic Snake game implementation for human players.
Use arrow keys to control the snake and try to eat food to grow longer
while avoiding walls and your own body.

Controls:
    - Arrow Keys: Change snake direction
    - Close Window: Quit game
"""

import pygame
import random
from enum import Enum
from collections import namedtuple

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
SPEED = 20       # Game speed (FPS) - slower for human players

class SnakeGame:
    """
    Classic Snake Game for Human Players
    
    A traditional Snake game where players control the snake using arrow keys.
    The goal is to eat food to grow longer while avoiding collisions with
    walls and the snake's own body.
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
        
        # Initialize game state
        self.direction = Direction.RIGHT
        
        # Create snake starting at center with 3 segments
        self.head = Point(self.w/2, self.h/2)
        self.snake = [
            self.head, 
            Point(self.head.x - BLOCK_SIZE, self.head.y),
            Point(self.head.x - (2*BLOCK_SIZE), self.head.y)
        ]
        
        # Initialize score and place first food
        self.score = 0
        self.food = None
        self._place_food()
        
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
        
    def play_step(self):
        """
        Execute one step of the game loop.
        
        This method:
        1. Processes user input (keyboard events)
        2. Moves the snake in the current direction
        3. Checks for collisions (game over conditions)
        4. Handles food consumption and score updates
        5. Updates the display
        
        Returns:
            tuple: (game_over, score)
                - game_over (bool): True if the game ended
                - score (int): Current score
        """
        # 1. Collect user input from keyboard
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
            if event.type == pygame.KEYDOWN:
                # Update direction based on arrow key pressed
                if event.key == pygame.K_LEFT:
                    self.direction = Direction.LEFT
                elif event.key == pygame.K_RIGHT:
                    self.direction = Direction.RIGHT
                elif event.key == pygame.K_UP:
                    self.direction = Direction.UP
                elif event.key == pygame.K_DOWN:
                    self.direction = Direction.DOWN
        
        # 2. Move the snake in the current direction
        self._move(self.direction)  # Update head position
        self.snake.insert(0, self.head)  # Add new head to snake
        
        # 3. Check if game over (collision detected)
        game_over = False
        if self._is_collision():
            game_over = True
            return game_over, self.score
            
        # 4. Check if snake ate food or just moved
        if self.head == self.food:
            # Snake ate food - increase score and spawn new food
            self.score += 1
            self._place_food()
        else:
            # Normal move - remove tail segment (snake doesn't grow)
            self.snake.pop()
        
        # 5. Update the display and control game speed
        self._update_ui()
        self.clock.tick(SPEED)
        
        # 6. Return game state
        return game_over, self.score
    
    def _is_collision(self):
        """
        Check if the snake has collided with walls or itself.
        
        Returns:
            bool: True if collision detected, False otherwise
        """
        # Check collision with boundaries
        if self.head.x > self.w - BLOCK_SIZE or self.head.x < 0 or self.head.y > self.h - BLOCK_SIZE or self.head.y < 0:
            return True
            
        # Check collision with snake's own body (excluding head)
        if self.head in self.snake[1:]:
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
        
    def _move(self, direction):
        """
        Update the snake's head position based on the direction.
        
        Args:
            direction: Direction enum value (RIGHT, LEFT, UP, or DOWN)
        """
        # Calculate new head position based on direction
        x = self.head.x
        y = self.head.y
        if direction == Direction.RIGHT:
            x += BLOCK_SIZE
        elif direction == Direction.LEFT:
            x -= BLOCK_SIZE
        elif direction == Direction.DOWN:
            y += BLOCK_SIZE
        elif direction == Direction.UP:
            y -= BLOCK_SIZE
            
        self.head = Point(x, y)
            

if __name__ == '__main__':
    """
    Main entry point for the human-playable Snake game.
    
    Initializes the game and runs the main game loop until the player
    collides with a wall or the snake's body.
    """
    # Create a new game instance
    game = SnakeGame()
    
    # Main game loop
    while True:
        game_over, score = game.play_step()
        
        # Exit loop when game ends
        if game_over == True:
            break
    
    # Display final score
    print('Final Score', score)
        
    # Clean up and close Pygame
    pygame.quit()