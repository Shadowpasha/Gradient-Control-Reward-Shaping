import gymnasium as gym
import numpy as np
import time
import argparse

# Import variants
from mobile_robot import MobileRobotEnv as BaseEnv
from mobile_robot_GCR import MobileRobotGCREnv as GCREnv
from mobile_robot_PBRS import MobileRobotPBRSEnv as PBRSEnv
from mobile_robot_PBRS_Bias import MobileRobotPBRSBiasEnv as PBRSBiasEnv
from mobile_robot_BSRS import MobileRobotBSRSEnv as BSRSEnv

def main():
    parser = argparse.ArgumentParser(description="View an example episode of the Mobile Robot environment.")
    parser.add_argument("--env", type=str, default="standard", 
                        choices=["standard", "gcr", "pbrs", "pbrs_bias", "bsrs"],
                        help="The variant of the environment to view.")
    parser.add_argument("--episodes", type=int, default=3, help="Number of episodes to run.")
    args = parser.parse_args()

    # Environment Selection
    if args.env == "standard":
        env = BaseEnv(render_mode="human")
    elif args.env == "gcr":
        env = GCREnv(render_mode="human")
    elif args.env == "pbrs":
        env = PBRSEnv(render_mode="human")
    elif args.env == "pbrs_bias":
        env = PBRSBiasEnv(render_mode="human")
    elif args.env == "bsrs":
        env = BSRSEnv(render_mode="human")
    else:
        print("Invalid environment choice.")
        return

    print(f"Viewing environment: {args.env}")
    print("Close the viewer window to stop.")

    try:
        for ep in range(args.episodes):
            obs, info = env.reset()
            done = False
            total_reward = 0
            steps = 0
            
            print(f"Episode {ep + 1} starting...")
            
            while not done:
                # Use a simple random action (mostly forward, slight turn)
                action = np.array([0.5, np.random.uniform(-0.1, 0.1)])
                
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                
                env.render()
                
                total_reward += reward
                steps += 1
                
                # Maintain ~20 FPS
                time.sleep(1.0 / 20.0)
                
                if terminated:
                    if info.get("collision"):
                        print("CRASHED!")
                    else:
                        print("GOAL REACHED!")
                
            print(f"Episode finished. Steps: {steps}, Total Reward: {total_reward:.2f}")
            time.sleep(1.0) # Pause between episodes
            
    except KeyboardInterrupt:
        print("Visualization stopped by user.")
    finally:
        env.close()
        print("Environment closed.")

if __name__ == "__main__":
    main()
