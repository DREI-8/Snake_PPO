import torch
import torch.nn as nn
import numpy

class Critic():
    """
    Value function approximator (Critic) for PPO algorithm.
    Estimates the value of being in a particular state.
    """
    def __init__(self, env, args):
        self.env = env
        self.args = args
        self.obs_dim = env.observation_space.shape[0]
        # self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = torch.device("cpu")
        self.model = self.build_model().to(self.device)
        self.eval()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=args["critic_lr"])

    def build_model(self):
        """
        Builds a neural network for value function approximation.
        Architecture: state_dim -> 64 -> 64 -> 1
        """
        return nn.Sequential(
            nn.Linear(self.obs_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1)  # Single output for state value
        )

    def predict(self, state, tensor=False):
        """
        Predicts the value of a state.
        Args:
            state: Current environment state
            tensor: Whether to return a tensor or a numpy array
        Returns:
            value: Estimated state value
        """
        state = torch.FloatTensor(numpy.array(state)).to(self.device)
        if tensor: # Used for training, requires gradient
            value = self.model(state)
        else: # Used for evaluation, no gradient needed
            with torch.no_grad():
                value = self.model(state)
           
        if tensor:
            return value
        return value.item()

    def optimize(self, value_loss):
        """
        Updates the value function.
        Args:
            value_loss: Value loss to optimize
        """
        self.optimizer.zero_grad()
        value_loss.backward()
        self.optimizer.step()

    def save(self, path):
        """Saves the critic model to disk."""
        torch.save(self.model.state_dict(), path)

    def load(self, path):
        """Loads the critic model from disk."""
        self.model.load_state_dict(torch.load(path))
    
    def eval(self):
        """Sets the model to evaluation mode."""
        self.model.eval()

    def train(self):
        """Sets the model to training mode."""
        self.model.train()