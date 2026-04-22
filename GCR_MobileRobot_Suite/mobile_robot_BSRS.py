import gymnasium as gym
from gymnasium import spaces
import numpy as np
import mujoco
import os
import time
from typing import Optional, Union
import torch as th
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from stable_baselines3 import SAC, TD3

# Import base
from mobile_robot import MobileRobotEnv

class MobileRobotBSRSEnv(MobileRobotEnv):
    def __init__(self, 
                 render_mode: Optional[str] = None, 
                 model: Optional[Union[SAC, TD3]] = None,
                 eta: float = 1.0,
                 random_start: bool = False):
        super().__init__(render_mode=render_mode, random_start=random_start)
        self.gamma = 0.99
        self.eta = eta
        self.rl_model = model
        
        self.total_num_steps = 0
        self.prev_potential = 0.0

    def set_model(self, model):
        self.rl_model = model
        self.prev_potential = self._get_potential(self._get_obs())

    def _get_potential(self, obs: np.ndarray) -> float:
        """Estimates Phi(s) = eta * V(s) using the agent's critic."""
        if self.rl_model is None or not hasattr(self.rl_model, 'critic'):
            return 0.0
        
        obs_th = th.as_tensor(obs, device=self.rl_model.device).unsqueeze(0)
        with th.no_grad():
            # Get actions from actor
            if hasattr(self.rl_model, 'actor'):
                actions_th = self.rl_model.actor._predict(obs_th, deterministic=True)
            else:
                return 0.0

            # Get Q-values from critic
            q1, q2 = self.rl_model.critic(obs_th, actions_th)
            v_value = th.min(q1, q2).cpu().item()
            
        return self.eta * v_value

    def step(self, action):
        obs, reward_base, terminated, truncated, info = super().step(action)
        self.total_num_steps += 1
        
        # BSRS Shaping: F = gamma * Phi(s') - Phi(s)
        current_potential = self._get_potential(obs)
        shaping = self.gamma * current_potential - self.prev_potential
        self.prev_potential = current_potential
        
        # BSRS Shaping replaces dense reward, keeps terminal reward
        if terminated:
            reward = reward_base
        else:
            reward = shaping
        
        info['shaping_reward'] = shaping
        
        return obs, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)
        self.prev_potential = self._get_potential(obs)
        return obs, info

    def close(self):
        super().close()
