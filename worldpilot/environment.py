"""PyBullet drone environment wrapper for WorldPilot Stage 1.

The low-level quadrotor physics and velocity PID controller are provided by
gym-pybullet-drones. This wrapper adds randomized spherical obstacles,
waypoints, collision checks, and a compact state representation suitable
for learning a transition model.
"""
from __future__ import annotations

import time
import numpy as np
import pybullet as p

from gym_pybullet_drones.envs.VelocityAviary import VelocityAviary
from gym_pybullet_drones.utils.enums import DroneModel, Physics


class WorldPilotDroneEnv:
    """Single-CF2X drone in a randomized 3D obstacle course."""

    # state = position(3), velocity(3), rpy(3), goal-relative(3),
    #         four obstacle-relative records [dx,dy,dz,radius] (16)
    STATE_DIM = 28
    ACTION_DIM = 3  # commanded world-frame velocity in m/s

    def __init__(self, config, gui=False, seed=7):
        self.config = config
        self.gui = bool(gui)
        self.rng = np.random.default_rng(seed)
        sim = config["simulation"]
        world = config["world"]
        self.world = world
        self.ctrl_hz = int(sim["control_hz"])
        self.dt = 1.0 / self.ctrl_hz
        self.initial_position = np.asarray(world["initial_position"], dtype=np.float32)
        self.x_bounds = np.asarray(world["x_bounds"], dtype=np.float32)
        self.y_bounds = np.asarray(world["y_bounds"], dtype=np.float32)
        self.z_bounds = np.asarray(world["z_bounds"], dtype=np.float32)
        self.obstacle_count = int(world["obstacle_count"])
        self.goal_tolerance = float(world["goal_tolerance"])
        self.obstacle_ids = []
        self.obstacles = []
        self.goal = None
        self.goal_visual_id = None
        self.step_count = 0

        self.env = VelocityAviary(
            drone_model=DroneModel.CF2X,
            num_drones=1,
            initial_xyzs=self.initial_position.reshape(1, 3),
            initial_rpys=np.zeros((1, 3), dtype=np.float32),
            physics=Physics.PYB,
            pyb_freq=int(sim["physics_hz"]),
            ctrl_freq=self.ctrl_hz,
            gui=self.gui,
            record=False,
            obstacles=False,
            user_debug_gui=False,
        )
        self.client = self.env.getPyBulletClient()
        self.drone_id = self.env.getDroneIds()[0]
        # WorldPilot command limit, independent of the simulator default.
        self.max_speed = float(
            world.get("max_command_speed", self.env.SPEED_LIMIT)
        )

        # VelocityAviary uses this instance attribute when computing target velocity.
        self.env.SPEED_LIMIT = self.max_speed

        # Use the configured waypoint speed instead of a hardcoded value.
        self.waypoint_speed = float(
            config["collection"].get("waypoint_speed", self.max_speed)
        )

        if self.waypoint_speed <= 0 or self.waypoint_speed > self.max_speed:
            raise ValueError(
                "waypoint_speed must be positive and no greater than max_command_speed"
            )
        self.closed = False

        if self.gui:
            p.resetDebugVisualizerCamera(
                cameraDistance=6.5,
                cameraYaw=35,
                cameraPitch=-25,
                cameraTargetPosition=[0.0, 0.0, 0.8],
                physicsClientId=self.client,
            )

    def _remove_episode_objects(self):
        for body_id in self.obstacle_ids:
            try:
                p.removeBody(body_id, physicsClientId=self.client)
            except p.error:
                pass
        self.obstacle_ids = []
        self.obstacles = []
        if self.goal_visual_id is not None:
            try:
                p.removeBody(self.goal_visual_id, physicsClientId=self.client)
            except p.error:
                pass
            self.goal_visual_id = None

    def _sample_goal(self):
        for _ in range(100):
            candidate = np.array([
                self.rng.uniform(1.2, 2.5),
                self.rng.uniform(-2.1, 2.1),
                self.rng.uniform(0.65, 2.0),
            ], dtype=np.float32)
            if np.linalg.norm(candidate - self.initial_position) > 1.5:
                return candidate
        return np.array([2.0, 1.0, 1.0], dtype=np.float32)

    def _sample_obstacles(self):
        rmin, rmax = self.world["obstacle_radius_range"]
        obstacles = []
        attempts = int(self.world.get("obstacle_spawn_attempts", 100))
        for _ in range(self.obstacle_count):
            accepted = None
            for _attempt in range(attempts):
                radius = float(self.rng.uniform(rmin, rmax))
                center = np.array([
                    self.rng.uniform(-1.7, 2.0),
                    self.rng.uniform(-2.3, 2.3),
                    self.rng.uniform(0.55, 2.1),
                ], dtype=np.float32)
                # Keep a clear launch zone and avoid placing an obstacle on the goal.
                if np.linalg.norm(center - self.initial_position) < 1.0 + radius:
                    continue
                if np.linalg.norm(center - self.goal) < 0.85 + radius:
                    continue
                if any(np.linalg.norm(center - o["center"]) < radius + o["radius"] + 0.45
                       for o in obstacles):
                    continue
                accepted = {"center": center, "radius": radius}
                break
            if accepted is not None:
                obstacles.append(accepted)
        # Keep a fixed-size state vector: fill any unplaced obstacle slots far away.
        while len(obstacles) < self.obstacle_count:
            obstacles.append({
                "center": np.array([20.0 + len(obstacles), 20.0, 20.0], dtype=np.float32),
                "radius": 0.0,
            })
        return obstacles

    def _create_episode_objects(self):
        self.goal = self._sample_goal()
        self.obstacles = self._sample_obstacles()
        for obstacle in self.obstacles:
            center = obstacle["center"]
            radius = obstacle["radius"]
            if radius <= 0:
                continue
            collision = p.createCollisionShape(
                p.GEOM_SPHERE, radius=radius, physicsClientId=self.client
            )
            visual = p.createVisualShape(
                p.GEOM_SPHERE, radius=radius,
                rgbaColor=[0.90, 0.20, 0.16, 0.90],
                physicsClientId=self.client,
            )
            body = p.createMultiBody(
                baseMass=0,
                baseCollisionShapeIndex=collision,
                baseVisualShapeIndex=visual,
                basePosition=center.tolist(),
                physicsClientId=self.client,
            )
            self.obstacle_ids.append(body)

        goal_visual = p.createVisualShape(
            p.GEOM_SPHERE, radius=0.13,
            rgbaColor=[0.15, 0.85, 0.25, 0.75],
            physicsClientId=self.client,
        )
        self.goal_visual_id = p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=-1,
            baseVisualShapeIndex=goal_visual,
            basePosition=self.goal.tolist(),
            physicsClientId=self.client,
        )

    def reset(self, seed=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        # Remove episode objects BEFORE the aviary resets PyBullet. Resetting
        # PyBullet reuses body IDs, so removing old IDs afterwards could delete
        # newly-created ground/drone bodies.
        self._remove_episode_objects()
        obs, info = self.env.reset(seed=seed)
        self.drone_id = self.env.getDroneIds()[0]
        self._create_episode_objects()
        self.step_count = 0
        return self.get_state(obs[0]), info

    def get_state(self, raw_obs=None):
        if raw_obs is None:
            raw_obs = self.env._computeObs()[0]
        raw_obs = np.asarray(raw_obs, dtype=np.float32)
        pos = raw_obs[0:3]
        rpy = raw_obs[7:10]
        vel = raw_obs[10:13]
        goal_rel = self.goal - pos
        obstacle_features = []
        for obstacle in self.obstacles:
            obstacle_features.extend((obstacle["center"] - pos).tolist())
            obstacle_features.append(float(obstacle["radius"]))
        state = np.concatenate([pos, vel, rpy, goal_rel,
                                np.asarray(obstacle_features, dtype=np.float32)])
        if state.shape != (self.STATE_DIM,):
            raise RuntimeError(f"Expected state shape {(self.STATE_DIM,)}, got {state.shape}")
        return state.astype(np.float32)

    def action_to_env(self, velocity_command):
        """Convert m/s world-frame command to VelocityAviary's direction+speed action."""
        velocity = np.asarray(velocity_command, dtype=np.float32)
        norm = float(np.linalg.norm(velocity))
        if norm < 1e-7:
            return np.zeros((1, 4), dtype=np.float32)
        direction = velocity / norm
        speed_fraction = np.clip(norm / max(self.max_speed, 1e-6), 0.0, 1.0)
        return np.concatenate([direction, [speed_fraction]]).astype(np.float32).reshape(1, 4)

    def step(self, velocity_command):
        velocity_command = np.asarray(velocity_command, dtype=np.float32)
        # velocity_command = np.clip(velocity_command, -self.max_speed, self.max_speed)
        # Enforce a true 3D velocity-magnitude limit.
        command_speed = float(np.linalg.norm(velocity_command))

        if command_speed > self.max_speed:
            velocity_command = (
                velocity_command * (self.max_speed / command_speed)
            )
        env_action = self.action_to_env(velocity_command)
        raw_obs, reward, terminated, truncated, info = self.env.step(env_action)
        self.step_count += 1
        next_state = self.get_state(raw_obs[0])
        collision = self.check_collision()
        position = next_state[0:3]
        goal_reached = bool(np.linalg.norm(self.goal - position) <= self.goal_tolerance)
        out_of_bounds = bool(
            position[0] < self.x_bounds[0] or position[0] > self.x_bounds[1]
            or position[1] < self.y_bounds[0] or position[1] > self.y_bounds[1]
            or position[2] < self.z_bounds[0] or position[2] > self.z_bounds[1]
        )
        done = collision or goal_reached or out_of_bounds
        info = dict(info)
        info.update({
            "collision": collision,
            "goal_reached": goal_reached,
            "out_of_bounds": out_of_bounds,
            "goal_distance": float(np.linalg.norm(self.goal - position)),
        })
        return next_state, float(reward), bool(done), info

    def check_collision(self):
        for obstacle_id in self.obstacle_ids:
            contacts = p.getClosestPoints(
                bodyA=self.drone_id, bodyB=obstacle_id, distance=0.015,
                physicsClientId=self.client,
            )
            if contacts:
                return True
        return False

    def waypoint_controller(self, state, exploration_sigma=0.0, random_action_probability=0.0,
                            random_action_scale=0.5, influence_distance=1.15,
                            repulsion_gain=0.75):
        """Simple potential-field controller with optional exploratory perturbations."""
        pos = state[0:3]
        goal_rel = state[9:12]
        goal_vec = goal_rel / (np.linalg.norm(goal_rel) + 1e-8)
        # desired = 0.48 * goal_vec
        desired = self.waypoint_speed * goal_vec

        # Obstacle features start at index 12; each record is relative xyz + radius.
        for idx in range(self.obstacle_count):
            offset = 12 + 4 * idx
            rel = state[offset:offset + 3]
            radius = float(state[offset + 3])
            if radius <= 0:
                continue
            center_distance = float(np.linalg.norm(rel))
            surface_distance = center_distance - radius
            if 1e-5 < surface_distance < influence_distance:
                away = -rel / (center_distance + 1e-8)
                strength = repulsion_gain * (1.0 / max(surface_distance, 0.08)
                                             - 1.0 / influence_distance)
                desired += away * min(strength, 1.0)

        if self.rng.random() < random_action_probability:
            desired = self.rng.uniform(-random_action_scale, random_action_scale, size=3)
        elif exploration_sigma > 0:
            desired += self.rng.normal(0.0, exploration_sigma, size=3)

        norm = float(np.linalg.norm(desired))
        if norm > self.max_speed:
            desired = desired / norm * self.max_speed
        return desired.astype(np.float32)

    def draw_trajectory_segment(self, start, end, color=(0.1, 0.4, 1.0), width=2.0,
                                lifetime=0.0):
        if not self.gui:
            return
        p.addUserDebugLine(
            start.tolist(), end.tolist(), lineColorRGB=list(color),
            lineWidth=width, lifeTime=lifetime, physicsClientId=self.client,
        )

    def close(self):
        if not self.closed:
            self.env.close()
            self.closed = True
