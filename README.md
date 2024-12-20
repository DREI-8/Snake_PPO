# Snake PPO

A pedagogical Proximal Policy Optimization (PPO) project applied to the classic Snake game.

## Overview
This repository demonstrates how an agent, based on the PPO algorithm, learns to play Snake by collecting food and avoiding collisions. Over time, the agent discovers a strategy of quickly circling around the reward before taking it to minimize self-collisions, especially since it does not precisely track its entire tail position.

- The environment is implemented in [`Game/Snake.py`](Game/Snake.py).
- The PPO agent logic is in [`Agent/PPO.py`](Agent/PPO.py).
- Usage examples and training procedure are shown in [`main.ipynb`](main.ipynb).

## Installation
1. Clone this repository.  
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
    ```
3. You can then run or modify [`main.ipynb`](main.ipynb) to train or test the PPO agent.

## PPO Algorithm
PPO is a policy gradient method designed to stabilize training by limiting updates to the policy. In this project:
- We use clipping (controlled by `"clip_epsilon"`) to avoid overly large updates.
- We incorporate entropy regularization (controlled by `"entropy_coef"`) to encourage exploration.
- We apply Generalized Advantage Estimation (GAE) for more stable advantage computation.

The core training code is in the `train` function of [`PPOAgent`](Agent/PPO.py), and the environment loops in [`SnakeEnv`](Game/Snake.py).

## Demo
Below is a demo of the trained agent playing, occasionally circling around the reward to avoid self-collisions (it doesn't know its tail's exact position, see [`main.ipynb`](main.ipynb) for the observation space details):

![Demo GIF placeholder](/assets/demo_agent_10000.gif)

## Results
The agent’s reward curve increases as it masters collecting food. Episode length first rises with better survival but eventually decreases when it chooses to sacrifice longevity for quicker gains:

![Reward Curves placeholder](/assets/rewards_curves_10000.png)

## Usage
- Train the agent by running:
  ```python
  # Inside main.ipynb
  agent.train(total_epochs=10000, steps_per_epoch=4096)
    ```
- Test the trained agent (with optional rendering):
  ```python
  agent.test_episode(render=True)
    ```

Explore [`main.ipynb`](main.ipynb) for more details on experiments, and review the in-code comments for deeper understanding.