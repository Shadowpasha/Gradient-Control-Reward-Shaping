
import gymnasium as gym
import numpy as np
from mobile_robot import MobileRobotEnv

def test_env():
    env = MobileRobotEnv()
    obs, info = env.reset()
    print(f"Robot ID: {env.root_id}, Robot ADR: {env.root_qposadr}")
    print(f"Initial Pos: {env.data.qpos[env.root_qposadr : env.root_qposadr + 2]}")
    
    for i in range(10):
        # Move forward (max v)
        action = np.array([1.0, 0.0]) 
        obs, reward, terminated, truncated, info = env.step(action)
        pos = env.data.qpos[env.root_qposadr : env.root_qposadr + 2].copy()
        print(f"Step {i+1}: Reward={reward:.4f}, Pos={pos}, NormDist={obs[20]:.4f}")
        if terminated or truncated:
            break
            
    env.close()

if __name__ == "__main__":
    test_env()
