import matplotlib.pyplot as plt
import pandas as pd

def plot_metrics(metrics):
    """
    Display training metrics.
    Args:
        metrics: Dictionary containing training metrics
            - episode_returns: List of episode returns
            - episode_lengths: List of episode lengths
            - value_losses: List of value function losses
            - policy_losses: List of policy losses
            - entropie_losses: List of entropy values
    """
    # Create figure with subplots
    fig, axs = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Training Metrics', size=16, weight='bold')
    
    # Calculate rolling means for smoother curves
    window = 40
    returns_smooth = pd.Series(metrics['episode_returns']).rolling(window).mean()
    lengths_smooth = pd.Series(metrics['episode_lengths']).rolling(window).mean()
    
    # Plot episode returns
    ax = axs[0, 0]
    ax.plot(metrics['episode_returns'], alpha=0.3, color='#2ecc71', label='Raw')
    ax.plot(returns_smooth, color='#27ae60', label=f'Rolling mean ({window})')
    ax.set_title('Episode Returns', weight='bold')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Return')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Plot episode lengths
    ax = axs[0, 1]
    ax.plot(metrics['episode_lengths'], alpha=0.3, color='#3498db', label='Raw')
    ax.plot(lengths_smooth, color='#2980b9', label=f'Rolling mean ({window})')
    ax.set_title('Episode Lengths', weight='bold')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Length')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Plot losses
    ax = axs[1, 0]
    ax.plot(metrics['value_losses'], color='#e74c3c', label='Value Loss', alpha=0.8)
    ax.plot(metrics['policy_losses'], color='#9b59b6', label='Policy Loss', alpha=0.8)
    ax.set_title('Training Losses', weight='bold')
    ax.set_xlabel('Update Step')
    ax.set_ylabel('Loss')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Plot entropy
    ax = axs[1, 1]
    ax.plot(metrics['entropie_losses'], color='#f1c40f', label='Entropy Loss', alpha=0.8)
    ax.set_title('Entropy Loss', weight='bold')
    ax.set_xlabel('Update Step')
    ax.set_ylabel('Entropy')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Adjust layout and display
    plt.tight_layout()
    plt.show()
