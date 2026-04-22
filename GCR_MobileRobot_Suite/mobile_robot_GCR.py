import gymnasium as gym
from gymnasium import spaces
import numpy as np
import mujoco
import os
import time
from typing import Optional
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

# Import base
from mobile_robot import MobileRobotEnv
from gcr import GCR

class MobileRobotGCREnv(MobileRobotEnv):
    def __init__(self, render_mode: Optional[str] = None, random_start: bool = False):
        super().__init__(render_mode=render_mode, random_start=random_start)
        
        self.total_num_steps = 0
        
        # Initialize GCR with requested gains
        # Kp=0.2, Ki=0.05, Kd=0.0005, Ku=0.1, Ke=1.0
        self.gcr = GCR(Kp=0.17, Ki=0.05, Kd=0.0005, Ku=0.1, Ke=1.0, 
                       output_limits=(-2.5, 2.5), 
                       integeral_limits=(-0.45, 0.45))
        
        self.prev_time = time.time_ns()

    def step(self, action):
        # Call super().step() to get the base logic
        # We'll then override the reward with GCR if not terminal
        obs, reward_base, terminated, truncated, info = super().step(action)
        
        dist = obs[20] * 10.0 
        
        # GCR Logic
        curr_time = time.time_ns()
        dt = (curr_time - self.prev_time) / 1e9
        if dt <= 0: dt = 1e-6
        
        gcr_reward = self.gcr.update(setpoint=0.0, measurement=dist, sample_time=dt)
        self.prev_time = curr_time
        
        # GCR replaces the dense progress reward
        if terminated:
            reward = reward_base  # Contains +100 for Goal or -50 for Collision
        else:
            reward = gcr_reward   # Pure GCR dense reward
            
        # Debugging and logging
        info['gcr_reward'] = gcr_reward
        if self.step_counter % 50 == 0:
            print(f"Step: {self.step_counter} | Dist: {dist:.3f} | BaseRew (Sparse): {reward_base if terminated else 0:.3f} | GCRRew: {gcr_reward:.3f} | Total: {reward:.3f}")
            
        return obs, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)
        self.gcr.reset()
        self.prev_time = time.time_ns()
        return obs, info

    def close(self):
        super().close()
