import os
import re
import numpy as np
from tensorboard.backend.event_processing import event_accumulator

def get_best_run(log_dir, variant_name):
    """Find the best run for a given variant in the tensorboard directory."""
    best_value = -1.0
    best_run_dir = None
    best_seed = None
    
    # regex to match variant in folder name
    # e.g. mobile_GCR_1, gcr_seed_1234_1
    pattern = re.compile(rf".*{variant_name}.*", re.IGNORECASE)
    
    for run_name in os.listdir(log_dir):
        if not pattern.match(run_name):
            continue
            
        run_path = os.path.join(log_dir, run_name)
        if not os.path.isdir(run_path):
            continue
            
        try:
            # Read tensorboard events
            ea = event_accumulator.EventAccumulator(run_path)
            ea.Reload()
            
            tags = ea.Tags().get('scalars', [])
            tag = "performance/success_rate_last_10"
            
            if tag in tags:
                events = ea.Scalars(tag)
                if events:
                    # Get the max success rate in this run
                    max_val = max([e.value for e in events])
                    if max_val > best_value:
                        best_value = max_val
                        best_run_dir = run_name
                        
                        # Try to extract seed from name if available
                        seed_match = re.search(r"seed_(\d+)", run_name)
                        if seed_match:
                            best_seed = seed_match.group(1)
                        else:
                            best_seed = None
        except Exception as e:
            print(f"Error reading {run_name}: {e}")
            
    return best_run_dir, best_value, best_seed

def get_best_models(log_dir="./tensorboard", models_dir="./models"):
    variants = {
        "Standard": "Standard",
        "GCR": "GCR",
        "PBRS": "PBRS",
        "PBRS_Bias": "PBRS_Bias",
        "BSRS": "BSRS"
    }
    best_models = {}
    
    print(f"Scanning {log_dir} for best models...")
    for var_key, var_name in variants.items():
        best_run, best_val, best_seed = get_best_run(log_dir, var_name)
        if best_run:
            print(f"  Best {var_key}: {best_run} (Success Rate: {best_val:.2f})")
            
            # Try to find the model file
            model_path = None
            if best_seed:
                # Look for model with seed
                potential_path = os.path.join(models_dir, "seeds", f"{var_name.lower()}_seed_{best_seed}.zip")
                if os.path.exists(potential_path):
                    model_path = potential_path.replace(".zip", "")
            
            if not model_path:
                # Fallback to default name
                potential_path = os.path.join(models_dir, f"mobile_sac_{var_name.lower()}.zip")
                if os.path.exists(potential_path):
                    model_path = potential_path.replace(".zip", "")
            
            if model_path:
                best_models[var_key] = model_path
                
    return best_models

if __name__ == "__main__":
    best = get_best_models()
    print("\nBest models found:")
    for k, v in best.items():
        print(f"  {k}: {v}")
