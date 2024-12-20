import pygame
import numpy as np
import gymnasium as gym
from gymnasium import spaces

class SnakeEnv(gym.Env):
    """
    A Snake game environment following the OpenAI Gym interface.
    
    Parameters
    ----------
    - width (int): Width of the game window in pixels
    - height (int): Height of the game window in pixels
    - grid_size (int): Size of each grid cell in pixels
    - render_mode (str): Rendering mode ('human' for visual display, None for no rendering)
    """
    def __init__(self, width=640, height=480, grid_size=20, render_mode=None):
        super().__init__()
        self.width = width
        self.height = height
        self.grid_size = grid_size
        self.render_mode = render_mode
        
        self.action_space = spaces.Discrete(4)  # 4 possible actions
        
        # Whether to use intermediate rewards when the snake gets closer to the food
        self.intermediate_rewards = False

        # Observation space includes:
        # - Head position (2)
        # - Food position (2)
        # - Current direction (2)
        # - Snake vision in 8 directions (8)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(14,), dtype=np.float32
        )
        
        if render_mode == "human":
            pygame.init()
            self.display = pygame.display.set_mode((width, height))
            pygame.display.set_caption('Snake Game')
        else:
            self.display = None
            
        self.reset()
    
    def reset(self, seed=None):
        """
        Reset the environment to initial state.
        
        Parameters
        ----------
        - seed (int, optional): Random seed for reproducibility
            
        Returns
        -------
            tuple: (observation, info dictionary)
        """
        super().reset(seed=seed)
        
        head_pos = (self.width // 2, self.height // 2)
        second_pos = (head_pos[0] - self.grid_size, head_pos[1])
        self.snake = [head_pos, second_pos]
        self.direction = (self.grid_size, 0)
        self.score = 0
        self.food = self._place_food()
        self.steps = 0
        self.max_steps = 1000  # Step limit per episode

        if self.render_mode == "human" and self.display is None:
            pygame.init()
            self.display = pygame.display.set_mode((self.width, self.height))
            pygame.display.set_caption('Snake Game')
        
        return self._get_obs(), {}
    
    def _get_obs(self):
        """
        Get current observation state.
        
        Returns
        -------
            numpy.ndarray: Normalized observation vector containing head position,
                         food position, direction, and vision data
        """
        head_x, head_y = self.snake[0]
        food_x, food_y = self.food
        
        # Normalize positions
        norm_head_x = head_x / self.width
        norm_head_y = head_y / self.height
        norm_food_x = food_x / self.width
        norm_food_y = food_y / self.height
        
        # Normalized direction
        dir_x = self.direction[0] / self.grid_size
        dir_y = self.direction[1] / self.grid_size
        
        # Vision in 8 directions
        vision = self._get_vision()
        
        return np.array([
            norm_head_x, norm_head_y,
            norm_food_x, norm_food_y,
            dir_x, dir_y,
            *vision
        ], dtype=np.float32)
    
    def _get_vision(self):
        """
        Calculate distances to obstacles in all 8 directions.
        
        Returns
        -------
            list: Normalized distances to obstacles in 8 directions
        """
        directions = [
            (-1, 0), (1, 0), (0, -1), (0, 1),
            (-1, -1), (-1, 1), (1, -1), (1, 1)
        ]
        vision = []
        head_x, head_y = self.snake[0]
        max_distance = max(self.width, self.height) // self.grid_size

        for dx, dy in directions:
            distance = 1.0
            x, y = head_x, head_y
            while True:
                x += dx * self.grid_size
                y += dy * self.grid_size
                if (x, y) in self.snake or \
                   x < 0 or x >= self.width or \
                   y < 0 or y >= self.height:
                    break
                distance += 1
            vision.append(1.0 / min(distance, max_distance))
        return vision
    
    def _place_food(self):
        """Places food randomly on the grid, avoiding the snake's body.

        Returns
        -------
            tuple: A pair of (x, y) coordinates representing the food's position on the grid.
                The coordinates are multiples of grid_size to align with the grid.
        """
        while True:
            food = (np.random.randint(0, self.width // self.grid_size) * self.grid_size,
                    np.random.randint(0, self.height // self.grid_size) * self.grid_size)
            if food not in self.snake:
                return food
    
    def step(self, action):
        """
        Execute one time step within the environment.
        
        Parameters
        ----------
        - action (int): Action to take (0: Up, 1: Down, 2: Left, 3: Right)
            
        Returns
        -------
            tuple: (observation, reward, terminated, truncated, info)
        """
        self.steps += 1
        
        # Handle direction
        # Prevent snake from reversing
        if action == 0 and self.direction[1] != self.grid_size:  # Up
            self.direction = (0, -self.grid_size)
        elif action == 1 and self.direction[1] != -self.grid_size:  # Down
            self.direction = (0, self.grid_size)
        elif action == 2 and self.direction[0] != self.grid_size:  # Left
            self.direction = (-self.grid_size, 0)
        elif action == 3 and self.direction[0] != -self.grid_size:  # Right
            self.direction = (self.grid_size, 0)
            
        new_head = (self.snake[0][0] + self.direction[0],
                   self.snake[0][1] + self.direction[1])
        
        # Check for game over conditions
        done = (
            new_head in self.snake or
            new_head[0] < 0 or new_head[0] >= self.width or
            new_head[1] < 0 or new_head[1] >= self.height or
            self.steps >= self.max_steps
        )
        
        if done:
            reward = -10 # Large penalty for game over
            return self._get_obs(), reward, True, False, {}
        
        reward = -0.01  # Small penalty to encourage finding food quickly

        # Add intermediate reward if enabled
        if self.intermediate_rewards:
            # Calculate distance to food before move
            old_dist = ((self.snake[0][0] - self.food[0])**2 + 
                        (self.snake[0][1] - self.food[1])**2)**0.5
            # Calculate distance to food after move
            new_dist = ((new_head[0] - self.food[0])**2 + 
                        (new_head[1] - self.food[1])**2)**0.5
            
            if new_dist < old_dist:
                reward += 0.1  # Small reward for getting closer

        self.snake.insert(0, new_head)
        
        if new_head == self.food:
            self.score += 1
            self.food = self._place_food()
            reward = 10.0 # Large reward for finding food
        else:
            self.snake.pop()
        
        if self.render_mode == "human":
            self.render()
            
        return self._get_obs(), reward, False, False, {"score": self.score}
    
    def add_intermediate_rewards(self, enable=True):
        """
        Enable intermediate rewards when the snake gets closer to the food.
        
        Parameters
        ----------
        - enable (bool): Whether to enable intermediate rewards
        """
        self.intermediate_rewards = enable

    def render(self):
        """
        Render the game state visually.
        Only works when render_mode is set to 'human'.
        """
        if self.render_mode != "human" or self.display is None:
            return
            
        # Dark grid background
        self.display.fill((20, 20, 30))
        
        for x in range(0, self.width, self.grid_size):
            for y in range(0, self.height, self.grid_size):
                pygame.draw.rect(self.display, (30, 30, 40), 
                    (x, y, self.grid_size, self.grid_size), 1)

        # Draw snake with gradient
        for i, segment in enumerate(self.snake):
            if i == 0:  # Head
                color = (0, 255, 100)
                pygame.draw.rect(self.display, color, 
                    (*segment, self.grid_size, self.grid_size))
                # Eyes
                eye_size = self.grid_size // 4
                eye_offset = self.grid_size // 4
                pygame.draw.circle(self.display, (255, 255, 255),
                    (segment[0] + eye_offset, segment[1] + eye_offset), eye_size)
                pygame.draw.circle(self.display, (255, 255, 255),
                    (segment[0] + 3*eye_offset, segment[1] + eye_offset), eye_size)
            else:  # Body
                green = max(50, 255 - i * 20)
                pygame.draw.rect(self.display, (0, green, 0),
                    (*segment, self.grid_size, self.grid_size))
        
        # Draw food
        food_pos = (self.food[0] + self.grid_size//2, 
                self.food[1] + self.grid_size//2)
        pygame.draw.circle(self.display, (255, 0, 0), 
            food_pos, self.grid_size//2)
        
        pygame.display.flip()

    def change_render_mode(self, mode):
        """
        Change the rendering mode.
        
        Parameters
        ----------
        - mode (str): New rendering mode ('human' for visual display, None for no rendering)
        """
        if mode == "human":
            pygame.init()
            self.display = pygame.display.set_mode((self.width, self.height))
            pygame.display.set_caption('Snake Game')
        else:
            self.display = None
            pygame.quit()

        self.render_mode = mode
        
    def close(self):
        """
        Close the display.
        """
        if self.display is not None:
            pygame.quit()

    
    def play_human(self):
        """
        Allow human to play the game with visual feedback of environment observations.
        
        Controls:
        - Arrow keys: Control snake direction
        - Escape: Quit game
        """
        if self.render_mode != "human":
            raise ValueError("Play mode requires render_mode='human'")
            
        # Setup display with additional space for stats
        stats_width = 300
        pygame.display.set_mode((self.width + stats_width, self.height))
        
        # Font setup
        pygame.font.init()
        font = pygame.font.Font(None, 24)
        
        clock = pygame.time.Clock()
        running = True
        fps = 15
        
        obs, _ = self.reset()
        reward = 0
        needs_update = True  # Flag to track if display needs updating
        episode_terminated = False
        last_reward = 0
        
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                        
            
            keys = pygame.key.get_pressed()
            action = None
            if keys[pygame.K_UP]:
                action = 0
            elif keys[pygame.K_DOWN]:
                action = 1
            elif keys[pygame.K_LEFT]:
                action = 2
            elif keys[pygame.K_RIGHT]:
                action = 3
                
            if action is not None:
                if episode_terminated:
                    obs, _ = self.reset()
                    reward = 0
                    episode_terminated = False
                    needs_update = True
                else:
                    # Check if the action would make the snake go backwards
                    dx = self.direction[0]
                    dy = self.direction[1]
                    valid_action = True
                    
                    if (action == 0 and dy == self.grid_size) or \
                       (action == 1 and dy == -self.grid_size) or \
                       (action == 2 and dx == self.grid_size) or \
                       (action == 3 and dx == -self.grid_size):
                        valid_action = False
                    
                    if valid_action:
                        obs, reward, terminated, truncated, _ = self.step(action)
                        needs_update = True
                        if terminated or truncated:
                            episode_terminated = True
                            last_reward = reward
            
            clock.tick(fps)
            
            if needs_update:
                # Render game state
                self.display.fill((20, 20, 30))
                
                # Draw game
                self.render()
                
                # Draw stats background
                stats_rect = pygame.Surface((stats_width, self.height))
                stats_rect.fill((30, 30, 40))
                self.display.blit(stats_rect, (self.width, 0))
                
                # Draw stats
                stats = [
                    ("Score", self.score),
                    ("Steps", f"{self.steps}/{self.max_steps}"),
                    ("Reward", f"{reward:.2f}"),
                    "",
                    "Observations:",
                    f"Head pos: ({obs[0]:.2f}, {obs[1]:.2f})",
                    f"Food pos: ({obs[2]:.2f}, {obs[3]:.2f})",
                    f"Direction: ({obs[4]:.2f}, {obs[5]:.2f})",
                    "",
                    "Vision distances:",
                    "Left  : {:.2f}".format(obs[6]),
                    "Right : {:.2f}".format(obs[7]),
                    "Up    : {:.2f}".format(obs[8]),
                    "Down  : {:.2f}".format(obs[9]),
                    "DiagLU: {:.2f}".format(obs[10]),
                    "DiagLD: {:.2f}".format(obs[11]),
                    "DiagRU: {:.2f}".format(obs[12]),
                    "DiagRD: {:.2f}".format(obs[13]),
                ]
                
                y_offset = 20
                for stat in stats:
                    if isinstance(stat, tuple):
                        text = font.render(f"{stat[0]}: {stat[1]}", True, (200, 200, 200))
                    else:
                        text = font.render(stat, True, (200, 200, 200))
                    self.display.blit(text, (self.width + 20, y_offset))
                    y_offset += 25
                
                if episode_terminated:
                    term_text = font.render(f"Episode terminated, last reward = {last_reward}", True, (255, 0, 0))
                    self.display.blit(term_text, (20, self.height - 40))
                    
                pygame.display.flip()
                needs_update = False
        
        pygame.quit()
        self.display = None
