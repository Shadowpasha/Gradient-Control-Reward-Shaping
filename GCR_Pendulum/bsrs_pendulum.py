__credits__ = ["Carlos Luis", "Gymnasium Authors"]

from os import path
from typing import Optional, Union

import numpy as np
import torch as th
from datetime import datetime

import gymnasium as gym
from gymnasium import spaces
from gymnasium.envs.classic_control import utils
from gymnasium.error import DependencyNotInstalled

# Import for type hinting and TensorBoard logging
try:
    from stable_baselines3 import SAC, TD3
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    class SAC: pass
    class TD3: pass
    class SummaryWriter:
        def __init__(self, log_dir=None): print("Warning: TensorBoard not found. Logging is disabled.")
        def add_scalar(self, *args, **kwargs): pass
        def close(self): pass
    print("Warning: stable-baselines3 or torch not found. Type hinting/logging may be limited.")


DEFAULT_X = np.pi
DEFAULT_Y = 1.0


class PendulumEnvWithBSRS(gym.Env):
    """
    Pendulum Environment with integrated Bootstrapped Reward Shaping (BSRS)
    and monitoring for upright time.
    """
    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 30,
    }

    def __init__(self,
                 render_mode: Optional[str] = None,
                 g=10.0,
                 model: Optional[Union[SAC, TD3]] = None,
                 gamma: float = 0.99,
                 eta: float = 1.0,
                 enable_logging: bool = True):
        self.max_speed = 8
        self.max_torque = 2.0
        self.dt = 0.05
        self.g = g
        self.m = 1.0
        self.l = 1.0

        self.model = model
        self.gamma = gamma
        self.eta = eta
        self._last_potential = 0.0

        self.render_mode = render_mode
        self.screen_dim = 500
        self.screen = None
        self.clock = None
        self.isopen = True

        high = np.array([1.0, 1.0, self.max_speed], dtype=np.float32)
        self.action_space = spaces.Box(
            low=-self.max_torque, high=self.max_torque, shape=(1,), dtype=np.float32
        )
        self.observation_space = spaces.Box(low=-high, high=high, dtype=np.float32)

        self._elapsed_steps = 0

        # --- ADDED FOR MONITORING ---
        self.enable_logging = enable_logging
        if self.enable_logging:
            log_dir = "runs/upright_time_bsrs_" + datetime.now().strftime("%m_%d_%Y_%H_%M_%S")
            self.writer = SummaryWriter(log_dir)
        self.total_num_steps = 0
        self.episode_upright_time = 0
        # --- END ADDITION ---


    def set_model(self, model: Union[SAC, TD3]):
        """Sets the model for BSRS after the environment has been initialized."""
        if model is None:
             raise ValueError("A model instance (SAC or TD3) must be provided.")
        if not hasattr(model, 'critic') or not hasattr(model, 'actor'):
             raise ValueError("The provided model must have 'critic' and 'actor' attributes.")
        self.model = model
        # Re-initialize potential if needed (usually handled in reset)
        self._last_potential = 0.0

    def _get_potential(self, obs: np.ndarray) -> float:
        """Estimates the potential Phi(s) = eta * V(s) using the agent's critic."""
        if self.model is None:
            return 0.0
        obs_th = th.as_tensor(obs, device=self.model.device).unsqueeze(0)
        with th.no_grad():
            if isinstance(self.model, (SAC, TD3)):
                 actions_th = self.model.actor._predict(obs_th, deterministic=True)
            else:
                 actions_np, _ = self.model.actor.predict(obs_th.cpu().numpy(), deterministic=True)
                 actions_th = th.as_tensor(actions_np, device=self.model.device)

            q_values_1, q_values_2 = self.model.critic(obs_th, actions_th)
            q_values_min = th.min(q_values_1, q_values_2)
            v_value = q_values_min.cpu().item()
        return self.eta * v_value

    def step(self, u):
        th, thdot = self.state
        g, m, l, dt = self.g, self.m, self.l, self.dt
        u = np.clip(u, -self.max_torque, self.max_torque)[0]
        self.last_u = u

        costs = angle_normalize(th) ** 2 + 0.1 * thdot**2 + 0.001 * (u**2)
        reward_original = -costs

        newthdot = thdot + (3 * g / (2 * l) * np.sin(th) + 3.0 / (m * l**2) * u) * dt
        newthdot = np.clip(newthdot, -self.max_speed, self.max_speed)
        newth = th + newthdot * dt
        self.state = np.array([newth, newthdot])

        self._elapsed_steps += 1
        terminated = False
        truncated = self._elapsed_steps >= 200

        # --- ADDED UPRIGHT TIME MONITORING ---
        if(abs(angle_normalize(self.state[0])) < 0.0872  and abs(self.state[1]) < 0.1):
            self.episode_upright_time += 1
        # --- END ADDITION ---

        prev_potential = self._last_potential
        current_obs = self._get_obs()
        current_potential = self._get_potential(current_obs)
        shaping_reward_F = self.gamma * current_potential - prev_potential
        shaped_reward = float(reward_original + shaping_reward_F)
        self._last_potential = current_potential
        
        info = {'original_reward': reward_original}

        # --- ADDED LOGGING ON EPISODE END ---
        if truncated or terminated:
            self.total_num_steps += self._elapsed_steps
            if self.enable_logging:
                self.writer.add_scalar('reward/upright_time', self.episode_upright_time, self.total_num_steps)
        # --- END ADDITION ---

        if self.render_mode == "human":
            self.render()

        return current_obs, shaped_reward, terminated, truncated, info

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        high = np.array([np.pi, 1])
        self.state = self.np_random.uniform(low=-high, high=high)
        self.last_u = None
        self._elapsed_steps = 0

        # --- ADDED RESET FOR UPRIGHT TIME ---
        self.episode_upright_time = 0
        # --- END ADDITION ---

        initial_obs = self._get_obs()
        self._last_potential = self._get_potential(initial_obs)

        if self.render_mode == "human":
            self.render()
        return initial_obs, {}

    def _get_obs(self):
        theta, thetadot = self.state
        return np.array([np.cos(theta), np.sin(theta), thetadot], dtype=np.float32)

    def render(self):
        if self.render_mode is None: return
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
            else:
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
        theta = self.state[0]
        for c in coords:
            c = pygame.math.Vector2(c).rotate_rad(theta + np.pi / 2)
            c = (c[0] + offset, c[1] + offset)
            transformed_coords.append(c)
        gfxdraw.aapolygon(self.surf, transformed_coords, (204, 77, 77))
        gfxdraw.filled_polygon(self.surf, transformed_coords, (204, 77, 77))
        gfxdraw.aacircle(self.surf, offset, offset, int(rod_width / 2), (204, 77, 77))
        gfxdraw.filled_circle(self.surf, offset, offset, int(rod_width / 2), (204, 77, 77))
        rod_end = (rod_length, 0)
        rod_end = pygame.math.Vector2(rod_end).rotate_rad(theta + np.pi / 2)
        rod_end = (int(rod_end[0] + offset), int(rod_end[1] + offset))
        gfxdraw.aacircle(self.surf, rod_end[0], rod_end[1], int(rod_width / 2), (204, 77, 77))
        gfxdraw.filled_circle(self.surf, rod_end[0], rod_end[1], int(rod_width / 2), (204, 77, 77))
        gfxdraw.aacircle(self.surf, offset, offset, int(0.05 * scale), (0, 0, 0))
        gfxdraw.filled_circle(self.surf, offset, offset, int(0.05 * scale), (0, 0, 0))
        self.surf = pygame.transform.flip(self.surf, False, True)
        self.screen.blit(self.surf, (0, 0))

        if self.render_mode == "human":
            pygame.event.pump()
            self.clock.tick(self.metadata["render_fps"])
            pygame.display.flip()
        elif self.render_mode == "rgb_array":
            return np.transpose(np.array(pygame.surfarray.pixels3d(self.screen)), axes=(1, 0, 2))

    def close(self):
        if self.screen is not None:
            import pygame
            pygame.display.quit()
            pygame.quit()
            self.isopen = False
        # Close the TensorBoard writer
        if self.enable_logging and hasattr(self, 'writer'):
            self.writer.close()

def angle_normalize(x):
    return ((x + np.pi) % (2 * np.pi)) - np.pi

