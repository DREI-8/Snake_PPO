import torch
import torch.nn as nn
from torch.distributions import Categorical
import numpy

class Policy():
    """
    Policy network for PPO algorithm.
    Implements a neural network that maps states to action probabilities.
    """
    def __init__(self, env, args):
        self.env = env
        self.args = args
        self.obs_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.n
        # self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = torch.device("cpu")
        self.model = self.build_model().to(self.device)
        self.eval()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=args["policy_lr"])

    def build_model(self):
        """
        Builds a neural network for the policy.
        Architecture: state_dim -> 64 -> 64 -> action_dim
        """
        return nn.Sequential(
            nn.Linear(self.obs_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, self.action_dim),
            nn.Softmax(dim=-1)
        )

    def sample_action(self, state):
        """
        Samples an action from the policy.
        Args:
            state: Current environment state
        Returns:
            action: Selected action
            action_proba: Probability of selected action
        """
        state = torch.FloatTensor(state).to(self.device)
        with torch.no_grad(): # Used for evaluation, no gradient needed
            action_probs = self.model(state)
        
        # Create a distribution and sample from it
        dist = Categorical(action_probs)
        action = dist.sample()

        return action.item(), action_probs[action.item()].item()
    def proba(self, state, action):
        """
        Computes the probability of taking an action in a state and the entropy loss.
        Args:
            state: Environment state
            action: Action to evaluate
        Returns:
            tuple: (probability of the action, entropy loss)
        """
        state = torch.FloatTensor(numpy.array(state)).to(self.device)
        action = torch.LongTensor(numpy.array(action)).to(self.device)
        action_probs = self.model(state)
        
        # Calculate entropy loss
        dist = Categorical(action_probs)
        entropy_loss = -self.args["entropy_coef"] * dist.entropy().mean()
        
        if action.dim() == 1:
            action = action.unsqueeze(-1)
        
        action_probability = action_probs.gather(1, action).squeeze(-1)

        return action_probability, entropy_loss
    
    def optimize(self, loss_clip):
        """
        Optimize the policy using the clipped loss.
        Args:
            loss_clip: Clipped loss
        """
        self.optimizer.zero_grad()
        loss_clip.backward()
        self.optimizer.step()
    
    def save(self, path):
        """Save the policy model."""
        torch.save(self.model.state_dict(), path)
    
    def load(self, path):
        """Load the policy model."""
        self.model.load_state_dict(torch.load(path))
    
    def eval(self):
        """Set the policy model to evaluation mode."""
        self.model.eval()
    
    def train(self):
        """Set the policy model to training mode."""
        self.model.train()
        