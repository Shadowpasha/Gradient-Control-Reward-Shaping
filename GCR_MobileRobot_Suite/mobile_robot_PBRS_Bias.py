import gymnasium as gym
from gymnasium import spaces
import numpy as np
import mujoco
import os
import time
from typing import Optional
from torch.utils.tensorboard import SummaryWriter

# Import base
from mobile_robot import MobileRobotEnv

class MobileRobotPBRSBiasEnv(MobileRobotEnv):
    def __init__(self, render_mode: Optional[str] = None, bias: float = 0.05, random_start: bool = False):
        super().__init__(render_mode=render_mode, random_start=random_start)
        self.gamma = 0.99
        self.bias = bias # Bias term from the paper
        
        self.total_num_steps = 0
        self.prev_potential = 0.0

    def _get_potential(self, dist):
        # Biased potential logic
        base_potential = max(0.0, 1.0 - dist / 10.0)
        return base_potential + self.bias / (self.gamma - 1.0)

    def step(self, action):
        obs, reward_base, terminated, truncated, info = super().step(action)
        self.total_num_steps += 1
        
        dist = obs[20] * 10.0
        
        # PBRS-Bias Shaping
        current_potential = self._get_potential(dist)
        shaping = self.gamma * current_potential - self.prev_potential
        self.prev_potential = current_potential
        
        # PBRS-Bias Shaping replaces dense reward, keeps terminal reward
        if terminated:
            reward = reward_base
        else:
            reward = shaping
        
        return obs, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)
        self.prev_potential = self._get_potential(obs[20] * 10.0)
        return obs, info

    def close(self):
        super().close()
