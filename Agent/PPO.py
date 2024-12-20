from Agent.Policy import Policy
from Agent.Critic import Critic
import numpy as np
import torch
import time
from tqdm.notebook import tqdm
import os

class PPOAgent():
    """
    Proximal Policy Optimization (PPO) agent.
    Implements the PPO algorithm with:
    - GAE (Generalized Advantage Estimation)
    - Value function clipping
    - Entropy bonus for exploration
    """
    def __init__(self, env, args):
        self.env = env
        # The policy is a neural network that maps observations to action probabilities
        self.policy = Policy(env, args)
        # The critic is a neural network that estimates the value of being in a particular state
        self.critic = Critic(env, args)
        self.args = args

        self.metrics = {
            'episode_returns': [],
            'episode_lengths': [],
            'value_losses': [],
            'policy_losses': [],
            'entropie_losses': []
        }

    def collect_experience(self, num_steps):
        """
        Collect trajectory data for training.
        Args:
            num_steps: Number of environment steps to collect
        """
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.action_probs = []
        self.dones = []

        state, _ = self.env.reset()
        done = False

        current_episode_return = 0
        current_episode_length = 0

        self.policy.eval()
        self.critic.eval()
        for _ in range(num_steps):
            # Get action and value
            action, action_prob = self.policy.sample_action(state)
            value = self.critic.predict(state)
            
            # Take step in environment
            next_state, reward, done, truncated, _ = self.env.step(action)
            done = done or truncated
            
            current_episode_return += reward
            current_episode_length += 1

            # Store transition
            self.states.append(state)
            self.actions.append(action)
            self.rewards.append(reward)
            self.values.append(value)
            self.action_probs.append(action_prob)
            self.dones.append(done)
            
            state = next_state
            if done:
                # Store episode metrics when episode ends
                self.metrics['episode_returns'].append(current_episode_return)
                self.metrics['episode_lengths'].append(current_episode_length)
                # Reset episode tracking
                current_episode_return = 0
                current_episode_length = 0
                state, _ = self.env.reset()
    
    def compute_advantage(self, next_value):
        """
        Compute Generalized Advantage Estimation (GAE).
        Args:
            next_value: Value estimate for the state after the last state in trajectory
        Returns:
            advantages: GAE advantages
            returns: Discounted returns
        """
        advantages = []
        gae = 0

        # Convert to numpy for faster computation
        values = np.append(self.values, next_value)
        rewards = np.array(self.rewards)
        dones = np.array(self.dones)

        # Compute GAE backwards
        for t in reversed(range(len(rewards))):
            # Calculate Temporal Difference (TD) error
            # delta = reward(t) + gamma * V(s(t+1)) - V(s(t))
            # where V(s(t)) is the estimated value of the state at time t (sum of rewards from t to end)
            # It can be seen as the advantage of taking an action at time t
            delta = rewards[t] + self.args["gamma"] * values[t+1] * (1 - dones[t]) - values[t]
            # Compute GAE
            # GAE(t) = delta + gamma * lambda * GAE(t+1)
            # - lambda close to 1 considers longer-term rewards but might introduce more bias.
            # - lambda close to 0 focuses only on immediate rewards, reducing variance but ignoring future contributions.
            gae = delta + self.args["gamma"] * self.args["gae_lambda"] * (1 - dones[t]) * gae
            # Add GAE to the beginning of the list because we computed it backwards
            advantages.insert(0, gae)
            
        advantages = np.array(advantages)
        # Compute returns
        # Advantage(s,a) = Q(s,a) - V(s)
        # where Q(s,a) is the return of taking action a in state s (expected sum of rewards from t to end taking action a)
        returns = advantages + np.array(self.values)
    
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        return advantages, returns
    
    def optimize_policy(self, states, actions, advantages, old_probs):
        """
        Update policy using PPO clipping objective.
        The policy is updated to maximize the probability 
        of actions that have higher advantages and entropy, to encourage exploration.
        Args:
            states: Batch of states
            actions: Batch of actions
            advantages: Computed advantages
            old_probs: Action probabilities under old policy
        """
        advantages = torch.FloatTensor(advantages).to(self.policy.device)
        old_probs = torch.FloatTensor(old_probs).to(self.policy.device)
        
        # Set policy to training mode
        self.policy.train()

        # Get the action probabilities under the current policy and the entropy loss
        # - current_action_probs: The probability of taking the actual actions under current policy
        # - entropy_loss: Additional loss term to encourage policy exploration by maximizing action distribution entropy
        current_action_probs, entropy_loss = self.policy.proba(states, actions)
        
        # Compute ratio and clipped ratio
        ratio = current_action_probs / old_probs
        # See https://arxiv.org/abs/1707.06347 for details on clipping
        clipped_ratio = torch.clamp(ratio, 1 - self.args["clip_epsilon"], 1 + self.args["clip_epsilon"])
        
        # Compute losses
        policy_loss = -torch.min(ratio * advantages, clipped_ratio * advantages).mean()

        total_loss = policy_loss + entropy_loss
        
        self.policy.optimize(total_loss)
        return policy_loss.item(), entropy_loss.item()
    
    def optimize_value(self, states, returns):
        """
        Update value function.
        Args:
            states: Batch of states
            returns: Computed returns
        """
        returns = torch.FloatTensor(returns).to(self.critic.device)
        
        # Set critic to training mode
        self.critic.train()

        # Predict values
        values = self.critic.predict(states, tensor=True)
        
        # Compute value loss : MSE between predicted values and returns
        value_loss = ((values - returns.unsqueeze(1))**2).mean()
        
        self.critic.optimize(value_loss)
        return value_loss.item()
    
    def train(self, total_epochs, steps_per_epoch, auto_save=True):
        """
        Train the agent using PPO.
        Args:
            total_epochs: Number of training epochs
            steps_per_epoch: Number of environment steps per epoch
        """

        # Linearly decay learning rate
        self.policy_scheduler = torch.optim.lr_scheduler.LinearLR(
            self.policy.optimizer,
            start_factor=1.0,
            end_factor=0.1,
            total_iters=total_epochs
        )
        self.critic_scheduler = torch.optim.lr_scheduler.LinearLR(
            self.critic.optimizer,
            start_factor=1.0,
            end_factor=0.1,
            total_iters=total_epochs
        )

        # Train agent
        for epoch in tqdm(range(total_epochs), desc="Training Progress"):
            # Collect trajectory
            self.collect_experience(steps_per_epoch)
            
            # Get final value for GAE calculation
            if len(self.dones) > 0 and not self.dones[-1]:
                last_state = self.env.reset() if self.dones[-1] else self.states[-1]
                final_value = self.critic.predict(last_state)
            else:
                final_value = 0
                
            # Compute advantages and returns
            advantages, returns = self.compute_advantage(final_value)
            
            # Optimize policy and value function
            policy_loss, entropy_loss = self.optimize_policy(
                self.states, self.actions, advantages, self.action_probs
            )
            value_loss = self.optimize_value(self.states, returns)
            
            # Store metrics
            self.metrics['value_losses'].append(value_loss)
            self.metrics['policy_losses'].append(policy_loss)
            self.metrics['entropie_losses'].append(entropy_loss)
            
            # Update learning rate
            self.policy_scheduler.step()
            self.critic_scheduler.step()

            # Evaluate agent periodically
            if epoch % self.args["eval_interval"] == 0:
                self.test_episode(render=False)
                
            # Save model periodically
            if auto_save and epoch % self.args["save_interval"] == 0:
                self.save(f"models/ppo_epoch_{epoch}")
        
        self.policy.eval()
        self.critic.eval()

    def test_episode(self, render=True, delay=0.05):
        """
        Test the agent on one episode.
        Args:
            render: Whether to render the environment
            delay: Time delay between steps for visualization
        Returns:
            total_reward: Total episode reward
            episode_length: Length of episode
        """
        if render:
            self.env.change_render_mode("human")
    
        state, _ = self.env.reset()
        total_reward = 0
        episode_length = 0
        done = False
        
        while not done:
            if render:
                self.env.render()
                time.sleep(delay)
                
            action, _ = self.policy.sample_action(state)
            state, reward, done, truncated, _ = self.env.step(action)
            done = done or truncated
            
            total_reward += reward
            episode_length += 1
        
        if render:
            self.env.change_render_mode(None)
            
        return total_reward, episode_length

    def save(self, path):
        """Save both policy and critic models."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.policy.save(f"{path}_policy.pth")
        self.critic.save(f"{path}_critic.pth")
        
    def load(self, path):
        """Load both policy and critic models."""
        self.policy.load(f"{path}_policy.pth")
        self.critic.load(f"{path}_critic.pth")
        self.policy.eval()
        self.critic.eval()