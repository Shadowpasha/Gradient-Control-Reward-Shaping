import gymnasium as gym
import numpy as np
import mujoco
import os
from typing import Optional
from mobile_robot import MobileRobotEnv

class MobileRobotValidationEnv(MobileRobotEnv):
    """
    A harder validation environment for the mobile robot.
    Features:
    - 8 randomized boxes (increased obstacle density).
    - Gaussian noise on LiDAR sensors.
    - Gaussian noise on robot state observations.
    - Slightly randomized physical parameters (actuator gain).
    """
    def __init__(self, render_mode: Optional[str] = None, random_start: bool = False):
        # Switch to the Box Forest layout
        self.xml_name = "mobile_robot_box_forest.xml"
        super().__init__(render_mode=render_mode, random_start=random_start)
        
        # Reload with box forest XML
        xml_path = os.path.join(os.path.dirname(__file__), "assets", self.xml_name)
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        
        # RE-INITIALIZE IDs for the new model instance
        self.root_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "root")
        self.root_qposadr = self.model.jnt_qposadr[self.root_id]
        self.root_qveladr = self.model.jnt_dofadr[self.root_id]
        self.robot_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "robot")
        self.floor_geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
        
        self.lidar_site_ids = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, f"s{i}") for i in range(20)]
        self.lidar_mocap_ids = []
        for i in range(20):
            beam_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, f"lidar_beam_{i}")
            if beam_id != -1:
                self.lidar_mocap_ids.append(self.model.body_mocapid[beam_id])
            else:
                self.lidar_mocap_ids.append(-1)
        
        # Noise parameters
        self.lidar_noise_std = 0.02    # 2% noise on LiDAR
        self.lidar_dropout_prob = 0.05 # 5% chance of beam dropout
        self.state_noise_std = 0.01    # 1% noise on robot state
        self.actuator_noise_std = 0.02 # 2% noise on actuators
        
        # Number of boxes for validation
        self.num_validation_boxes = 20 # Full forest!

    def _check_pos(self, x, y):
        # Boundary check only (Forest has no internal walls)
        if abs(x) > 4.5 or abs(y) > 4.5: return False
        return True

    def _get_obs(self):
        # Get base observation
        obs = super()._get_obs()
        
        # Add noise and dropout to LiDAR (first 20 elements)
        for i in range(20):
            if self.np_random.random() < self.lidar_dropout_prob:
                obs[i] = 1.0 # Simulate max range dropout
            else:
                noise = self.np_random.normal(0, self.lidar_noise_std)
                obs[i] = np.clip(obs[i] + noise, 0.0, 1.0)
        
        # Add noise to state (last 5 elements: dist, cos, sin, v, w)
        state_noise = self.np_random.normal(0, self.state_noise_std, size=5)
        obs[20:25] = obs[20:25] + state_noise
        
        return obs.astype(np.float32)

    def step(self, action):
        # Add actuator noise
        noisy_action = action + self.np_random.normal(0, self.actuator_noise_std, size=action.shape)
        noisy_action = np.clip(noisy_action, self.action_space.low, self.action_space.high)
        
        return super().step(noisy_action)

    def reset(self, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)
        
        # PHYSICS RANDOMIZATION
        # Randomize robot mass (+/- 20%)
        new_mass = self.np_random.uniform(4.0, 6.0)
        self.model.body_mass[self.robot_body_id] = new_mass
        
        # Randomize floor friction (+/- 30%)
        new_friction = self.np_random.uniform(0.7, 1.3)
        self.model.geom_friction[self.floor_geom_id, 0] = new_friction
        
        # After base reset handle boxes 0-2, we handle 3-19
        rx, ry = self.data.qpos[self.root_qposadr : self.root_qposadr + 2]
        gx, gy = self.goal_pos
        
        placed_objects = [(rx, ry), (gx, gy)]
        # Add the 3 boxes already placed by super().reset()
        for i in range(3):
            box_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, f"box{i}")
            box_qposadr = self.model.jnt_qposadr[self.model.body_jntadr[box_body_id]]
            bx, by = self.data.qpos[box_qposadr : box_qposadr + 2]
            placed_objects.append((bx, by))
            
        # Place extra boxes
        for i in range(3, self.num_validation_boxes):
            box_body_name = f"box{i}"
            box_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, box_body_name)
            box_joint_adr = self.model.body_jntadr[box_body_id]
            box_qposadr = self.model.jnt_qposadr[box_joint_adr]

            valid_box = False
            while not valid_box:
                bx = self.np_random.uniform(-4.2, 4.2)
                by = self.np_random.uniform(-4.2, 4.2)
                
                too_close = False
                for idx, (ox, oy) in enumerate(placed_objects):
                    dist = np.linalg.norm([bx-ox, by-oy])
                    clearance = 2.0 if idx == 0 else 1.2 
                    if dist < clearance:
                        too_close = True
                        break
                
                if not too_close:
                    valid_box = self._check_pos(bx, by)
            
            placed_objects.append((bx, by))
            self.data.qpos[box_qposadr : box_qposadr + 2] = [bx, by]
            self.data.qpos[box_qposadr + 2] = 0.3
            self.data.qpos[box_qposadr + 3 : box_qposadr + 7] = [1.0, 0.0, 0.0, 0.0]

        # Move unused boxes far away (up to 20 total)
        for i in range(self.num_validation_boxes, 20):
            box_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, f"box{i}")
            if box_body_id != -1:
                box_qposadr = self.model.jnt_qposadr[self.model.body_jntadr[box_body_id]]
                self.data.qpos[box_qposadr : box_qposadr + 3] = [10.0, 10.0, -1.0]
            
        mujoco.mj_forward(self.model, self.data)
        return self._get_obs(), {}
