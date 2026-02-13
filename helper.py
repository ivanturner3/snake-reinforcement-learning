"""
Helper Module for Training Visualization

This module provides real-time plotting functionality to visualize the
training progress of the reinforcement learning agent. It displays both
individual game scores and the running average score.
"""

import matplotlib.pyplot as plt
from IPython import display

# Enable interactive plotting mode for live updates
plt.ion()

def plot(scores, mean_scores):
    """
    Plot training progress in real-time.
    
    Creates a live-updating plot showing:
    - Individual game scores (blue line)
    - Running mean score (orange line)
    - Current values displayed as text annotations
    
    This function is called after each game to visualize learning progress.
    The plot updates dynamically, allowing you to see improvement over time.
    
    Args:
        scores: List of scores from all games played
        mean_scores: List of running average scores
    """
    # Clear previous output and display current figure
    display.clear_output(wait=True)
    display.display(plt.gcf())
    
    # Clear the current figure to prepare for new plot
    plt.clf()
    
    # Set plot title and axis labels
    plt.title('Training...')
    plt.xlabel('Number of Games')
    plt.ylabel('Score')
    
    # Plot the data
    plt.plot(scores, label='Score')          # Individual game scores
    plt.plot(mean_scores, label='Mean Score') # Running average
    
    # Set y-axis to start at 0 for better visualization
    plt.ylim(ymin=0)
    
    # Add text annotations showing the most recent values
    plt.text(len(scores)-1, scores[-1], str(scores[-1]))
    plt.text(len(mean_scores)-1, mean_scores[-1], str(mean_scores[-1]))
    
    # Display the plot without blocking execution
    plt.show(block=False)
    plt.pause(.1)  # Brief pause to allow plot to update