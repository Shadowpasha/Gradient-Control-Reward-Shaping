import gymnasium as gym
from gymnasium import spaces
import numpy as np
import mujoco
import os
from typing import Optional, Union

class MobileRobotEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 20}

    def __init__(self, render_mode: Optional[str] = None, random_start: bool = False):
        self.random_start = random_start
        # Path to the model file
        xml_path = os.path.join(os.path.dirname(__file__), "assets", "mobile_robot.xml")
        
        # Load MuJoCo model
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        
        self.render_mode = render_mode
        self.frame_skip = 50 
            
        self.total_num_steps = 0
        
        # Joint addresses
        self.root_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "root")
        self.root_qposadr = self.model.jnt_qposadr[self.root_id]
        self.root_qveladr = self.model.jnt_dofadr[self.root_id]
        
        # Action space: [v, w] where v is linear velocity [0, 3.0] and w is angular velocity [-1, 1]
        self.action_space = spaces.Box(low=np.array([0.0, -4.0]), high=np.array([4.0, 4.0]), dtype=np.float32)
        
        # Observation space: 20 (LiDAR) + 5 (normalized dist, cos(theta), sin(theta), v, w) = 25
        high = np.inf * np.ones(25, dtype=np.float32)
        self.observation_space = spaces.Box(low=-high, high=high, dtype=np.float32)
        
        # Cache IDs for LiDAR visualization
        self.lidar_site_ids = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, f"s{i}") for i in range(20)]
        self.lidar_mocap_ids = []
        for i in range(20):
            beam_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, f"lidar_beam_{i}")
            if beam_id != -1:
                self.lidar_mocap_ids.append(self.model.body_mocapid[beam_id])
            else:
                self.lidar_mocap_ids.append(-1)
        
        self.goal_pos = np.array([4.0, 0.0])
        self.prev_dist = 0.0
        self.step_counter = 0
        self.max_episode_steps = 700
        
        # Constants from reference script
        self.GOAL_REACHED_DIST = 0.5
        self.COLLISION_DIST = 0.4
        self.LIDAR_MAX_RANGE = 10.0
        
        # Success monitoring
        from collections import deque
        self.success_history = deque(maxlen=10)
        self.reached_goal = False

    def _get_obs(self):
        # 1. LiDAR data (20 rays)
        # rangefinder data is in self.data.sensordata. -1 means no hit.
        lidar_raw = self.data.sensordata.copy()
        lidar_raw[lidar_raw < 0] = self.LIDAR_MAX_RANGE # Map no-hit to max range
        lidar_data = lidar_raw / self.LIDAR_MAX_RANGE   # Normalize to [0, 1]
        
        # 2. Robot state
        pos = self.data.qpos[self.root_qposadr : self.root_qposadr + 2] # X, Y
        
        # Heading (yaw) from quaternion
        # yaw = atan2(2(wz + xy), 1 - 2(y^2 + z^2))
        q = self.data.qpos[self.root_qposadr + 3 : self.root_qposadr + 7] # w, x, y, z
        siny_cosp = 2 * (q[0] * q[3] + q[1] * q[2])
        cosy_cosp = 1 - 2 * (q[2] * q[2] + q[3] * q[3])
        yaw = np.arctan2(siny_cosp, cosy_cosp)
        
        # Distance to goal
        dist = np.linalg.norm(pos - self.goal_pos)
        
        # Theta (relative angle to goal)
        goal_vec = self.goal_pos - pos
        goal_angle = np.arctan2(goal_vec[1], goal_vec[0])
        theta = goal_angle - yaw
        
        # Normalize theta to [-pi, pi]
        theta = (theta + np.pi) % (2 * np.pi) - np.pi
        
        # Previous action (from actuator data or stored)
        prev_v = self.data.ctrl[0] 
        prev_w = self.data.ctrl[1] 
        
        # Normalized Distance to goal (Arena is 10x10)
        dist_norm = dist / 10.0
        
        # Theta representation: cos(theta), sin(theta)
        robot_state = np.array([dist_norm, np.cos(theta), np.sin(theta), prev_v, prev_w], dtype=np.float32)
        
        # 3. Update Lidar Visual Beams
        root_q = self.data.qpos[self.root_qposadr + 3 : self.root_qposadr + 7]
        for i in range(20):
            mocap_id = self.lidar_mocap_ids[i]
            if mocap_id == -1: continue
            
            site_id = self.lidar_site_ids[i]
            
            # Use MuJoCos built-in global site rotation matrix
            # The local Z-axis (ray forward) in global coords is the 3rd column
            direction = self.data.site_xmat[site_id].reshape(3, 3)[:, 2]
            
            dist_sensor = self.data.sensordata[i]
            if dist_sensor < 0: dist_sensor = self.LIDAR_MAX_RANGE
            
            self.data.mocap_pos[mocap_id] = self.data.site_xpos[site_id] + direction * dist_sensor
        
        return np.concatenate([lidar_data, robot_state]).astype(np.float32)

    def _check_pos(self, x, y):
        m = 0.6 # Increased margin for robot radius and safe spawning
        
        # L-Shape Bounds (Top Left)
        if -2.5 - m < x < -0.4 + m and 1.0 - m < y < 3.5 + m: return False
        
        # Hollow Square Bounds (Bottom Left)
        if -3.5 - m < x < -0.9 + m and -3.5 - m < y < -0.9 + m: return False
        
        # Thin Cross Bounds (Bottom Right)
        if 0.8 - m < x < 3.6 + m and -3.6 - m < y < -0.8 + m: return False
        
        # Hollow Triangle Bounds (Top Right)
        if 1.0 - m < x < 3.4 + m and 1.5 - m < y < 3.7 + m: return False
        
        # Boundary (Arena is +/- 5, wall inner face at +/- 4.9)
        if abs(x) > 4.5 or abs(y) > 4.5: return False
        return True

    def step(self, action):
        # Action is [v, w]
        v = action[0] # Forward only (0 to 2.0 m/s)
        w = action[1] # Scale to max 1.0 rad/s
        
        # Differential drive mapping to wheel velocities
        # v_L = v - w, v_R = v + w
        v_l = v - w
        v_r = v + w
        
        # Apply to actuators (ctrlrange -10 10 in XML)
        self.data.ctrl[0] = v_l # FL
        self.data.ctrl[1] = v_l # RL
        self.data.ctrl[2] = v_r # FR
        self.data.ctrl[3] = v_r # RR
        
        for _ in range(self.frame_skip):
            mujoco.mj_step(self.model, self.data)
            
        self.step_counter += 1
        self.total_num_steps += 1
        obs = self._get_obs()
        dist_norm = obs[20]
        dist = dist_norm * 10.0 # Re-scale for reward logic
        
        lidar_readings = obs[0:20] * self.LIDAR_MAX_RANGE
        valid_readings = lidar_readings[lidar_readings > 0]
        min_laser = np.min(valid_readings) if valid_readings.size > 0 else self.LIDAR_MAX_RANGE
        
        # Reward Logic Sync
        target_reached = dist < self.GOAL_REACHED_DIST
        collision = min_laser < self.COLLISION_DIST
        
        if target_reached:
            reward = 100.0
            self.reached_goal = True
        elif collision:
            reward = -50.0
        else:
            # Progress reward: (prev_dist - dist) * 10
            reward = (self.prev_dist - dist) * 10
            
        self.prev_dist = dist
        
        terminated = target_reached or collision
        truncated = self.step_counter >= self.max_episode_steps
        
        avg_success = 0.0
        if terminated or truncated:
            reason = "SUCCESS" if target_reached else ("COLLISION" if collision else "MAX_STEPS")
            print(f"Episode Done | Reason: {reason} | Steps: {self.step_counter} | Dist: {dist:.2f}m")
            self.success_history.append(1.0 if self.reached_goal else 0.0)
            avg_success = sum(self.success_history) / len(self.success_history) if self.success_history else 0.0
            
        return obs, reward, terminated, truncated, {"distance": dist, "success_rate": avg_success, "collision": collision}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)
        self.data.qvel[:] = 0 # Ensure zero velocity
        self.reached_goal = False
        self.total_num_steps = 0 if not hasattr(self, "total_num_steps") else self.total_num_steps
        
        # 1. Robot Position
        if self.random_start:
            valid_pos = False
            while not valid_pos:
                rx = self.np_random.uniform(-4.2, 4.2)
                ry = self.np_random.uniform(-4.2, 4.2)
                valid_pos = self._check_pos(rx, ry)
        else:
            rx, ry = 0.0, 0.0
        
        self.data.qpos[self.root_qposadr : self.root_qposadr + 2] = [rx, ry]
        self.data.qpos[self.root_qposadr + 2] = 0.08 # Spawn at wheel height
        
        # orientation
        if self.random_start:
            angle = self.np_random.uniform(-np.pi, np.pi)
        else:
            angle = 0.0 # Face X-axis
            
        self.data.qpos[self.root_qposadr + 3] = np.cos(angle/2) # w
        self.data.qpos[self.root_qposadr + 4] = 0.0 # x
        self.data.qpos[self.root_qposadr + 5] = 0.0 # y
        self.data.qpos[self.root_qposadr + 6] = np.sin(angle/2) # z
        
        # 2. Randomize Goal Position
        valid_goal = False
        while not valid_goal:
            gx = self.np_random.uniform(-4.2, 4.2)
            gy = self.np_random.uniform(-4.2, 4.2)
            if np.linalg.norm([gx-rx, gy-ry]) > 2.0: # Ensure goal isn't on top of robot
                valid_goal = self._check_pos(gx, gy)
        self.goal_pos = np.array([gx, gy])
        
        # Update goal position in MuJoCo (visual only)
        goal_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "goal")
        self.model.body_pos[goal_id] = [gx, gy, 0.1]
        
        # 3. Randomize Boxes (3 active)
        placed_objects = [(rx, ry), (gx, gy)] # Keep track of used space
        for i in range(3):
            box_body_name = f"box{i}"
            box_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, box_body_name)
            box_joint_adr = self.model.body_jntadr[box_body_id]
            box_qposadr = self.model.jnt_qposadr[box_joint_adr]

            valid_box = False
            while not valid_box:
                bx = self.np_random.uniform(-4.2, 4.2)
                by = self.np_random.uniform(-4.2, 4.2)
                
                # Check against robot, goal, and all existing boxes
                too_close = False
                for ox, oy in placed_objects:
                    if np.linalg.norm([bx-ox, by-oy]) < 1.5: # Increased clearance
                        too_close = True
                        break
                
                if not too_close:
                    valid_box = self._check_pos(bx, by)
            
            placed_objects.append((bx, by))
            self.data.qpos[box_qposadr : box_qposadr + 2] = [bx, by]
            self.data.qpos[box_qposadr + 2] = 0.3 # Box height
            self.data.qpos[box_qposadr + 3 : box_qposadr + 7] = [1.0, 0.0, 0.0, 0.0] # Upright
            
        mujoco.mj_forward(self.model, self.data) # Propagate geometric changes
        
        self.step_counter = 0
        self.prev_dist = np.linalg.norm(self.data.qpos[self.root_qposadr : self.root_qposadr + 2] - self.goal_pos)
        
        return self._get_obs(), {}

    def render(self):
        if self.render_mode == "human":
            if not hasattr(self, "viewer") or self.viewer is None:
                from mujoco import viewer
                self.viewer = viewer.launch_passive(self.model, self.data)
                # Polish: Turn off thick native sensor rays and enable our premium LiDAR fan
                with self.viewer.lock():
                    self.viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_RANGEFINDER] = 0
                    self.viewer.opt.geomgroup[1] = 1
                    self.viewer.opt.sitegroup[1] = 1
                    self.viewer.opt.tendongroup[1] = 1
            self.viewer.sync()

    def close(self):
        if hasattr(self, "viewer") and self.viewer is not None:
            self.viewer.close()
            self.viewer = None
