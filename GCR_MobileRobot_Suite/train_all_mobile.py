import gymnasium as gym
import numpy as np
import os
from stable_baselines3 import SAC
import argparse
import random
import torch as th

# Import environments
from mobile_robot import MobileRobotEnv
from mobile_robot_GCR import MobileRobotGCREnv
from mobile_robot_PBRS import MobileRobotPBRSEnv
from mobile_robot_PBRS_Bias import MobileRobotPBRSBiasEnv
from mobile_robot_BSRS import MobileRobotBSRSEnv
from stable_baselines3.common.callbacks import BaseCallback, CallbackList

class PerformanceLoggerCallback(BaseCallback):
    def __init__(self, render=False, verbose=0):
        super(PerformanceLoggerCallback, self).__init__(verbose)
        self.render = render

    def _on_step(self) -> bool:
        # Check if we have info from the environment
        if 'infos' in self.locals:
            info = self.locals['infos'][0]
            if 'success_rate' in info:
                self.logger.record("performance/success_rate_last_10", info['success_rate'])
            if 'distance' in info:
                self.logger.record("performance/distance_to_goal", info['distance'])
            if 'collision' in info:
                self.logger.record("performance/collision_occurred", float(info['collision']))
            if 'shaping_reward' in info:
                self.logger.record("performance/shaping_reward_bsrs", info['shaping_reward'])
            if 'gcr_reward' in info:
                self.logger.record("performance/gcr_reward", info['gcr_reward'])
        
        # Rendering
        if self.render:
            self.training_env.render()
            
        return True

class BSRSCallback(BaseCallback):
    """Update BSRS environment with the current model's critic."""
    def __init__(self, verbose=0):
        super(BSRSCallback, self).__init__(verbose)

    def _on_step(self) -> bool:
        # The environment is wrapped (Monitor, DummyVecEnv, etc.)
        # We need to find the underlying environment that has 'set_model'
        target_env = self.training_env.envs[0]
        while hasattr(target_env, 'env') and not hasattr(target_env, 'set_model'):
            target_env = target_env.env
            
        if hasattr(target_env, 'set_model'):
            target_env.set_model(self.model)
        else:
            # Fallback for SB3 env_method if direct access fails
            try:
                self.training_env.env_method('set_model', self.model, indices=0)
            except Exception:
                pass
        return True

def train_variant(env_class, name, timesteps=120000, render=False, random_start=False, seed=42):
    print(f"\n" + "="*50)
    print(f"Starting training: {name} (Render: {render}, RandomStart: {random_start}, Seed: {seed}, Steps: {timesteps})")
    print("="*50)
    
    # Reset global seeds before each variant for strict reproducibility
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        th.manual_seed(seed)
        if th.cuda.is_available():
            th.cuda.manual_seed_all(seed)

    # Initialize env with render mode and seed if requested
    render_mode = "human" if render else None
    env = env_class(render_mode=render_mode, random_start=random_start)
    
    # Set seed for environment
    if seed is not None:
        env.reset(seed=seed)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)

    # Match architecture from train_velodyne_node.py (800 -> 600)
    policy_kwargs = dict(net_arch=dict(pi=[800, 600], qf=[800, 600]))

    model = SAC("MlpPolicy", env, verbose=1, 
                seed=seed,
                learning_rate=3e-4,
                buffer_size=1000000,
                batch_size=40,
                gamma=0.99,
                tau=0.005,
                train_freq=1,
                gradient_steps=1,
                policy_kwargs=policy_kwargs,
                tensorboard_log="./tensorboard/")
    
    # Use CallbackList for multiple callbacks
    callbacks = [PerformanceLoggerCallback(render=render)]
    if name == "BSRS":
        callbacks.append(BSRSCallback())
        
    model.learn(total_timesteps=timesteps, callback=CallbackList(callbacks), tb_log_name=f"mobile_{name}")
    
    # Save model
    os.makedirs("models", exist_ok=True)
    model.save(f"models/mobile_sac_{name.lower()}")
    env.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train SAC agents on Mobile Robot environment variants.")
    parser.add_argument("--variant", type=str, default="all", 
                        choices=["all", "Standard", "GCR", "PBRS", "PBRS_Bias", "BSRS"],
                        help="The variant of the environment to train.")
    parser.add_argument("--timesteps", type=int, default=90000, help="Total timesteps to train.")
    parser.add_argument("--render", action="store_true", help="Enable environment rendering while training.")
    parser.add_argument("--random-start", action="store_true", help="Randomize the robot start position (default is fixed at 0,0).")
    parser.add_argument("--seed", type=int, default=553868, help="Seed for reproducibility.")
    args = parser.parse_args()

    variants_map = {
        "BSRS": (MobileRobotBSRSEnv, "BSRS"),
        "Standard": (MobileRobotEnv, "Standard"),
        "GCR": (MobileRobotGCREnv, "GCR"),
        "PBRS": (MobileRobotPBRSEnv, "PBRS"),
        "PBRS_Bias": (MobileRobotPBRSBiasEnv, "PBRS_Bias"),
    }
    
    if args.variant == "all":
        to_train = list(variants_map.values())
    else:
        to_train = [variants_map[args.variant]]
    
    for env_cls, name in to_train:
        try:
            train_variant(env_cls, name, timesteps=args.timesteps, render=args.render, random_start=args.random_start, seed=args.seed)
        except Exception as e:
            print(f"Error training {name}: {e}")
            import traceback
            traceback.print_exc()
