__credits__ = ["Carlos Luis", "Henrik Müller", "Daniel Kudenko"]

from os import path
from typing import Optional
from datetime import datetime

import numpy as np

import gymnasium as gym
from gymnasium import spaces
from gymnasium.envs.classic_control import utils
from gymnasium.error import DependencyNotInstalled
from torch.utils.tensorboard import SummaryWriter


DEFAULT_X = np.pi
DEFAULT_Y = 1.0


class ShapedPendulumEnv(gym.Env):
    """
    ## Description

    The inverted pendulum swingup problem with a shaped reward based on the paper
    "Improving the Effectiveness of Potential-Based Reward Shaping in Reinforcement Learning".

    The system consists of a pendulum attached at one end to a fixed point, and the other end being free.
    The pendulum starts in a random position and the goal is to apply torque on the free end to swing it
    into an upright position, with its center of gravity right above the fixed point.

    ## Action Space

    The action is a `ndarray` with shape `(1,)` representing the torque applied to free end of the pendulum.

    | Num | Action | Min  | Max |
    |-----|--------|------|-----|
    | 0   | Torque | -2.0 | 2.0 |

    ## Observation Space

    The observation is a `ndarray` with shape `(3,)` representing the x-y coordinates of the pendulum's free
    end and its angular velocity.

    | Num | Observation      | Min  | Max |
    |-----|------------------|------|-----|
    | 0   | x = cos(theta)   | -1.0 | 1.0 |
    | 1   | y = sin(theta)   | -1.0 | 1.0 |
    | 2   | Angular Velocity | -8.0 | 8.0 |

    ## Rewards

    The reward function is defined based on potential-based reward shaping. The potential
    is higher when the pendulum is upright. The shaping reward encourages the agent to move
    to states with higher potential.

    ## Starting State

    The starting state is a random angle in *[-pi, pi]* and a random angular velocity in *[-1,1]*.

    ## Episode Truncation

    The episode truncates at 200 time steps.
    """

    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 30,
    }

    def __init__(self, render_mode: Optional[str] = None, g=10.0, bias=0.0, enable_logging: bool = True):
        self.max_speed = 8
        self.max_torque = 2.0
        self.dt = 0.05
        self.g = g
        self.m = 1.0
        self.l = 1.0

        self.render_mode = render_mode
        self.bias = bias # The potential shift from the paper
        self.gamma = 0.99 # Discount factor, needed for PBRS

        # -- Logging for upright time --
        self.enable_logging = enable_logging
        if self.enable_logging:
            self.writer = SummaryWriter("runs/upright_time_potential_bias_" + datetime.now().strftime("%m_%d_%Y_%H_%M_%S"))
        self.total_num_steps = 0
        self.episode_upright_time = 0
        self.num_steps = 0
        # ----------------------------

        self.screen_dim = 500
        self.screen = None
        self.clock = None
        self.isopen = True

        high = np.array([1.0, 1.0, self.max_speed], dtype=np.float32)
        self.action_space = spaces.Box(
            low=-self.max_torque, high=self.max_torque, shape=(1,), dtype=np.float32
        )
        self.observation_space = spaces.Box(low=-high, high=high, dtype=np.float32)

    def _potential(self, state):
        """
        Potential function. Higher potential is better.
        The potential is based on the angle of the pendulum.
        Upright position (theta=0) has the highest potential.
        """
        theta, _ = state
        # We want the upright position (theta=0) to have the highest potential.
        # cos(theta) is 1 at theta=0 and -1 at theta=pi.
        # We can normalize this to be in [0, 1]
        potential = (np.cos(theta) + 1) / 2.0
        # Apply the bias shift from the paper
        return potential + self.bias / (self.gamma - 1)


    def step(self, u):
        th, thdot = self.state  # th := theta
        g = self.g
        m = self.m
        l = self.l
        dt = self.dt

        u = np.clip(u, -self.max_torque, self.max_torque)[0]
        self.last_u = u  # for rendering

        # Original cost function (negative reward)
        costs = angle_normalize(th) ** 2 + 0.1 * thdot**2 + 0.001 * (u**2)
        original_reward = -costs

        # --- Potential-Based Reward Shaping ---
        prev_potential = self._potential(self.state)

        # Dynamics
        newthdot = thdot + (3 * g / (2 * l) * np.sin(th) + 3.0 / (m * l**2) * u) * dt
        newthdot = np.clip(newthdot, -self.max_speed, self.max_speed)
        newth = th + newthdot * dt
        self.state = np.array([newth, newthdot])

        current_potential = self._potential(self.state)

        # Shaping reward F(s, a, s') = gamma * P(s') - P(s)
        shaping_reward = self.gamma * current_potential - prev_potential
        
        # Total reward is the original reward plus the shaping reward
        reward = original_reward + shaping_reward

        if self.render_mode == "human":
            self.render()

        # --- Tracking and Logging Upright Time ---
        terminated = False
        truncated = False
        
        # Check if the pendulum is in the upright position
        if(abs(angle_normalize(self.state[0])) < 0.0872  and abs(self.state[1]) < 0.1):
            self.episode_upright_time += 1
            
        self.num_steps += 1
        
        # Check if the episode is truncated
        if self.num_steps >= 200:
            self.total_num_steps += self.num_steps
            if self.enable_logging:
                self.writer.add_scalar('reward/upright_time', self.episode_upright_time, self.total_num_steps)
            truncated = True
        # ----------------------------------------

        return self._get_obs(), reward, terminated, truncated, {}

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        if options is None:
            high = np.array([DEFAULT_X, DEFAULT_Y])
        else:
            x = options.get("x_init", DEFAULT_X)
            y = options.get("y_init", DEFAULT_Y)
            high = np.array([x, y])

        low = -high  # Symmetric limits
        self.state = self.np_random.uniform(low=low, high=high)
        self.last_u = None
        
        # Reset episode-specific counters
        self.episode_upright_time = 0
        self.num_steps = 0

        if self.render_mode == "human":
            self.render()
        return self._get_obs(), {}

    def _get_obs(self):
        theta, thetadot = self.state
        return np.array([np.cos(theta), np.sin(theta), thetadot], dtype=np.float32)

    def render(self):
        if self.render_mode is None:
            gym.logger.warn("You are calling render method without specifying any render mode.")
            return

        try:
            import pygame
            from pygame import gfxdraw
        except ImportError:
            raise DependencyNotInstalled('pygame is not installed, run `pip install "gymnasium[classic_control]"`')

        if self.screen is None:
            pygame.init()
            if self.render_mode == "human":
                pygame.display.init()
                self.screen = pygame.display.set_mode((self.screen_dim, self.screen_dim))
            else:  # mode == "rgb_array"
                self.screen = pygame.Surface((self.screen_dim, self.screen_dim))
        if self.clock is None:
            self.clock = pygame.time.Clock()

        self.surf = pygame.Surface((self.screen_dim, self.screen_dim))
        self.surf.fill((255, 255, 255))

        bound = 2.2
        scale = self.screen_dim / (bound * 2)
        offset = self.screen_dim // 2

        rod_length = 1 * scale
        rod_width = 0.2 * scale
        l, r, t, b = 0, rod_length, rod_width / 2, -rod_width / 2
        coords = [(l, b), (l, t), (r, t), (r, b)]
        transformed_coords = []
        for c in coords:
            c = pygame.math.Vector2(c).rotate_rad(self.state[0] + np.pi / 2)
            c = (c[0] + offset, c[1] + offset)
            transformed_coords.append(c)
        gfxdraw.aapolygon(self.surf, transformed_coords, (204, 77, 77))
        gfxdraw.filled_polygon(self.surf, transformed_coords, (204, 77, 77))

        gfxdraw.aacircle(self.surf, offset, offset, int(rod_width / 2), (204, 77, 77))
        gfxdraw.filled_circle(self.surf, offset, offset, int(rod_width / 2), (204, 77, 77))

        rod_end = (rod_length, 0)
        rod_end = pygame.math.Vector2(rod_end).rotate_rad(self.state[0] + np.pi / 2)
        rod_end = (int(rod_end[0] + offset), int(rod_end[1] + offset))
        gfxdraw.aacircle(self.surf, rod_end[0], rod_end[1], int(rod_width / 2), (204, 77, 77))
        gfxdraw.filled_circle(self.surf, rod_end[0], rod_end[1], int(rod_width / 2), (204, 77, 77))

        fname = path.join(path.dirname(__file__), "assets/clockwise.png")
        if path.exists(fname):
            img = pygame.image.load(fname)
            if self.last_u is not None:
                scale_img = pygame.transform.smoothscale(
                    img, (scale * np.abs(self.last_u) / 2, scale * np.abs(self.last_u) / 2)
                )
                is_flip = bool(self.last_u > 0)
                scale_img = pygame.transform.flip(scale_img, is_flip, True)
                self.surf.blit(
                    scale_img,
                    (
                        offset - scale_img.get_rect().centerx,
                        offset - scale_img.get_rect().centery,
                    ),
                )

        gfxdraw.aacircle(self.surf, offset, offset, int(0.05 * scale), (0, 0, 0))
        gfxdraw.filled_circle(self.surf, offset, offset, int(0.05 * scale), (0, 0, 0))

        self.surf = pygame.transform.flip(self.surf, False, True)
        self.screen.blit(self.surf, (0, 0))
        if self.render_mode == "human":
            pygame.event.pump()
            self.clock.tick(self.metadata["render_fps"])
            pygame.display.flip()

        else:  # mode == "rgb_array"
            return np.transpose(np.array(pygame.surfarray.pixels3d(self.screen)), axes=(1, 0, 2))

    def close(self):
        if self.screen is not None:
            import pygame

            pygame.display.quit()
            pygame.quit()
            self.isopen = False
        if self.enable_logging and hasattr(self, 'writer'):
            self.writer.close()


def angle_normalize(x):
    return ((x + np.pi) % (2 * np.pi)) - np.pi

