import gymnasium as gym
import numpy as np
import os
import argparse
from stable_baselines3 import SAC
from mobile_robot_validation import MobileRobotValidationEnv

def evaluate_model(model_path, env, num_episodes=100):
    print(f"Evaluating {os.path.basename(model_path)}...")
    model = SAC.load(model_path)
    
    success_count = 0
    collision_count = 0
    total_steps = 0
    total_distance = 0.0
    
    for ep in range(num_episodes):
        obs, _ = env.reset()
        done = False
        ep_steps = 0
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            ep_steps += 1
            
            if hasattr(env, "render"):
                env.render()
            
            if terminated or truncated:
                done = True
                if info.get('success_rate', 0) > 0 or info.get('distance', 10.0) < 0.5: # Fallback success check
                    if not info.get('collision', False):
                        success_count += 1
                if info.get('collision', False):
                    collision_count += 1
                
                total_steps += ep_steps
                total_distance += info.get('distance', 0.0)
                
        if (ep + 1) % 10 == 0:
            print(f"  Episode {ep+1}/{num_episodes} | Successes: {success_count} | Collisions: {collision_count}")
            
    stats = {
        "success_rate": (success_count / num_episodes) * 100,
        "collision_rate": (collision_count / num_episodes) * 100,
        "avg_steps": total_steps / num_episodes,
        "avg_dist": total_distance / num_episodes
    }
    return stats

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate all mobile robot models in Hard Environment.")
    parser.add_argument("--episodes", type=int, default=100, help="Number of episodes per model.")
    parser.add_argument("--render", action="store_true", default=True, help="Render validation.")
    args = parser.parse_args()

    # Dynamically find the best models based on TensorBoard logs
    from find_best_models import get_best_models
    
    best_models_found = get_best_models()
    
    if not best_models_found:
        # Fallback to hardcoded defaults if log parsing fails or returns nothing
        print("Warning: Could not find best models from TensorBoard logs. Using defaults.")
        best_models_found = {
            "Standard": "models/mobile_sac_standard",
            "GCR": "models/mobile_sac_gcr",
            "PBRS": "models/mobile_sac_pbrs",
            "PBRS Bias": "models/mobile_sac_pbrs_bias",
            "BSRS": "models/mobile_sac_bsrs"
        }

    # Filter for existing models only
    existing_models = {name: path for name, path in best_models_found.items() if os.path.exists(path + ".zip")}
    
    if not existing_models:
        print("No models found. Please train them first.")
        exit(1)

    print(f"\nStarting Validation on {len(existing_models)} models for {args.episodes} episodes each.")
    print("Environment: Hard Mode (8 Boxes + Sensor Noise + Actuator Noise)")
    print("-" * 60)

    # Use a single environment instance
    env = MobileRobotValidationEnv(render_mode="human" if args.render else None, random_start=False)
    
    results = {}
    for name, path in existing_models.items():
        try:
            results[name] = evaluate_model(path, env, num_episodes=args.episodes)
        except Exception as e:
            print(f"Error evaluating {name}: {e}")

    env.close()

    # Print Final Summary Table
    print("\n" + "="*85)
    print(f"{'Model Variant':<20} | {'Success %':<12} | {'Collision %':<12} | {'Avg Steps':<12} | {'Avg Final Dist':<12}")
    print("-" * 85)
    for name, stats in results.items():
        print(f"{name:<20} | {stats['success_rate']:<12.1f} | {stats['collision_rate']:<12.1f} | {stats['avg_steps']:<12.1f} | {stats['avg_dist']:<12.2f}")
    print("="*85)
    print("\nValidation Complete. Results show performance under increased density and noise.")
