# Gradient Control Rewards for Reinforcement Learning

This repository contains the official implementation and reproducibility suite for the paper **"Gradient Control Rewards for Reinforcement Learning."**

The project focuses on **Gradient Control Reward (GCR)**, a novel reward shaping method designed to accelerate convergence and improve the robustness of Reinforcement Learning agents in control-oriented environments.

##  Overview

The codebase provides a comprehensive training and validation pipeline for a mobile robot navigation task. It compares GCR against several baseline reward shaping techniques:
- **Standard**: Sparse/Dense reward without shaping.
- **PBRS**: Potential-Based Reward Shaping.
- **PBRS-Bias**: PBRS with a bias term.
- **BSRS**: Boundary-Sensitive Reward Shaping.
- **GCR (Ours)**: Gradient Control Reward.

## Installation

### Prerequisites
- Python 3.10+
- MuJoCo 3.0+
- Gymnasium
- Stable-Baselines3

### Setup
```bash
# Clone the repository
git clone https://github.com/Shadowpasha/Gradient-Control-Reward-Shaping.git
cd Gradient-Control-Reward-Shaping

# Install dependencies
pip install gymnasium[mujoco] stable-baselines3 tensorboard
```

## 🏋️ Training Pipelines

The repository contains two main experiment suites:

### 1. Mobile Robot Navigation
Located in `GCR_MobileRobot_Suite/`. This environment tests agents in a 10x10m arena with obstacles.
```bash
cd GCR_MobileRobot_Suite
# Reproduce main results (Seed: 553868)
python3 train_all_mobile.py --seed 553868 --timesteps 90000
```

### 2. Classical Pendulum Swing-up
Located in `GCR_Pendulum/`. This environment evaluates GCR on the classic continuous control task.
```bash
cd GCR_Pendulum
# Reproduce all Pendulum results (Seed: 976458)
python3 train_all.py
```

## Validation & Hardening

The agents are evaluated in a "Hardened" environment to test generalization and robustness.

### Box Forest Validation
This mode places the agent in a dense forest of 20 static obstacles with sensor noise (LiDAR dropout) and actuator noise.
```bash
python3 validate_all_mobile.py --episodes 100 --render
```

### Environment Features:
- **Physics Randomization**: Randomized robot mass and floor friction.
- **Sensor Noise**: 5% LiDAR beam dropout and Gaussian observation noise.
- **High Density**: 20 obstacles in a 10x10m arena.
- **Dynamic Mode**: Optional dynamic drifting obstacles (forces applied to bodies).

## Monitoring Results

Monitor training progress and compare reward curves via TensorBoard:
```bash
tensorboard --logdir tensorboard/
```

## Project Structure

```text
├── GCR_MobileRobot_Suite/     # Mobile Robot experiments
│   ├── assets/                # MuJoCo XML models
│   ├── train_all_mobile.py    # Training entry point
│   └── validate_all_mobile.py # Hardened validation
├── GCR_Pendulum/              # Pendulum swing-up experiments
│   ├── train_all.py           # Training entry point
│   └── pendulum_GCR.py        # GCR Variant
├── README.md
```

