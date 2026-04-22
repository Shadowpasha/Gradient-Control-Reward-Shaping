import os
import gymnasium as gym
from stable_baselines3 import SAC
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.monitor import Monitor

# Import custom environments
from pendulum import PendulumEnv as BaseEnv
from pendulum_GCR import PendulumEnv as GCREnv
from pendulum_potential_based import PendulumEnv as PotentialBaseEnv
from Potential_bias import ShapedPendulumEnv as PotentialBiasEnv
from bsrs_pendulum import PendulumEnvWithBSRS as BSRSEnv

# --- Configuration ---
SEED = 976458
TOTAL_TIMESTEPS = 30000
TENSORBOARD_LOG = "./tensorboard/all_experiments/"
MODELS_DIR = "./models/all_experiments/"

# Setup directories
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(TENSORBOARD_LOG, exist_ok=True)

# Set global seed
set_random_seed(SEED)

def train_and_save(env, name, timesteps=TOTAL_TIMESTEPS):
    print(f"\n" + "="*50)
    print(f"Starting training: {name}")
    print("="*50)
    
    # Wrap environment with Monitor for better logging
    env = Monitor(env)
    
    # Instantiate the agent
    model = SAC(
        "MlpPolicy", 
        env, 
        verbose=1,
        seed=SEED,
        tensorboard_log=TENSORBOARD_LOG
    )
    
    # Train
    model.learn(total_timesteps=timesteps, tb_log_name=name)
    
    # Save
    save_path = os.path.join(MODELS_DIR, name)
    model.save(save_path)
    print(f"Finished: {name}. Model saved to {save_path}")
    
    # Close env
    env.close()

def main():
    # 1. Standard Pendulum
    train_and_save(BaseEnv(render_mode=None), "pendulum")
    
    # 2. GCR Pendulum
    train_and_save(GCREnv(render_mode=None), "gcr")
    
    # 3. Potential Base Pendulum
    train_and_save(PotentialBaseEnv(render_mode=None), "potential_base")
    
    # 4. Potential with Bias (using bias=0.5)
    train_and_save(PotentialBiasEnv(render_mode=None, bias=0.5), "potential_bias")
    
    # 5. BSRS Pendulum (Requires special initialization)
    print(f"\n" + "="*50)
    print("Starting training: bsrs")
    print("="*50)
    
    # Instantiate BSRSEnv first without a model
    # This avoids the circular dependency and the need for a dummy_env
    bsrs_env_raw = BSRSEnv(render_mode=None, model=None)
    bsrs_env = Monitor(bsrs_env_raw)
    
    # Create SAC model with the BSRS env
    bsrs_model = SAC(
        "MlpPolicy", 
        bsrs_env, 
        verbose=1,
        seed=SEED,
        tensorboard_log=TENSORBOARD_LOG
    )
    
    # Link the model back to the BSRS environment
    bsrs_env_raw.set_model(bsrs_model)
    
    # Link the model back to the Monitor/Env stack (already done via SAC init)
    
    # Train
    bsrs_model.learn(total_timesteps=TOTAL_TIMESTEPS, tb_log_name="bsrs")
    
    # Save
    bsrs_save_path = os.path.join(MODELS_DIR, "bsrs")
    bsrs_model.save(bsrs_save_path)
    print(f"Finished: bsrs. Model saved to {bsrs_save_path}")
    
    bsrs_env.close()

if __name__ == "__main__":
    main()
