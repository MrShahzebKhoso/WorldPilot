import numpy as np
import pybullet as p

from worldpilot.environment import WorldPilotDroneEnv


class IndoorMazeEnv:
    """Indoor grid maze built around the existing WorldPilot CF2X environment."""

    def __init__(self, base_config, maze_config, gui=True, seed=7):
        self.maze_config = maze_config["maze"]
        self.grid = self.maze_config["rows"]
        self.cell = float(self.maze_config["cell_size"])
        self.wall_height = float(self.maze_config["wall_height"])
        self.altitude = float(self.maze_config["flight_altitude"])
        self.start = tuple(self.maze_config["start"])
        self.goal_cell = tuple(self.maze_config["goal"])
        self.wall_ids = []
        self.closed = False

        rows, cols = len(self.grid), len(self.grid[0])
        if any(len(row) != cols for row in self.grid):
            raise ValueError("Maze rows must all have equal length")

        self.origin_x = -(cols - 1) * self.cell / 2.0
        self.origin_y = (rows - 1) * self.cell / 2.0

        start_xyz = self.cell_center(self.start)
        goal_xyz = self.cell_center(self.goal_cell)

        # Make a separate config: never modify the open-space experiment.
        config = {
            "simulation": dict(base_config["simulation"]),
            "world": dict(base_config["world"]),
            "collection": dict(base_config["collection"]),
        }
        config["simulation"]["gui"] = gui
        config["world"].update({
            "initial_position": start_xyz.tolist(),
            "x_bounds": [self.origin_x - self.cell, -self.origin_x + self.cell],
            "y_bounds": [-self.origin_y - self.cell, self.origin_y + self.cell],
            "z_bounds": [0.45, self.wall_height],
            "obstacle_count": 4,
            "obstacle_radius_range": [0.0, 0.0],
            "goal_tolerance": 0.28,
            "max_command_speed": 0.40,
        })
        config["collection"]["waypoint_speed"] = 0.35

        self.base = WorldPilotDroneEnv(config, gui=gui, seed=seed)
        self.client = self.base.client
        self.state, _ = self.base.reset(seed=seed)

        self.base.goal = goal_xyz.copy()

        # Replace the random goal marker created by the base environment.
        if self.base.goal_visual_id is not None:
            p.removeBody(self.base.goal_visual_id, physicsClientId=self.client)

        goal_visual = p.createVisualShape(
            p.GEOM_SPHERE, radius=0.18,
            rgbaColor=[0.1, 1.0, 0.2, 0.9],
            physicsClientId=self.client,
        )
        self.base.goal_visual_id = p.createMultiBody(
            baseMass=0,
            baseVisualShapeIndex=goal_visual,
            basePosition=goal_xyz.tolist(),
            physicsClientId=self.client,
        )

        self._build_walls()

    def cell_center(self, cell):
        row, col = cell
        x = self.origin_x + col * self.cell
        y = self.origin_y - row * self.cell
        return np.array([x, y, self.altitude], dtype=np.float32)

    def _build_walls(self):
        """Build thin wall panels along boundaries of blocked grid cells."""
        thickness = float(self.maze_config.get("wall_thickness", 0.15))
        half = self.cell / 2.0
        half_h = self.wall_height / 2.0

        vertical_collision = p.createCollisionShape(
            p.GEOM_BOX,
            halfExtents=[thickness / 2.0, half + thickness / 2.0, half_h],
            physicsClientId=self.client,
        )
        vertical_visual = p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=[thickness / 2.0, half + thickness / 2.0, half_h],
            rgbaColor=[0.55, 0.58, 0.63, 1.0],
            physicsClientId=self.client,
        )

        horizontal_collision = p.createCollisionShape(
            p.GEOM_BOX,
            halfExtents=[half + thickness / 2.0, thickness / 2.0, half_h],
            physicsClientId=self.client,
        )
        horizontal_visual = p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=[half + thickness / 2.0, thickness / 2.0, half_h],
            rgbaColor=[0.55, 0.58, 0.63, 1.0],
            physicsClientId=self.client,
        )

        rows, cols = len(self.grid), len(self.grid[0])

        # For every free cell, create a wall panel wherever its neighbor
        # is blocked or outside the map. Door openings remain completely open.
        for r in range(rows):
            for c in range(cols):
                if self.grid[r][c] == "#":
                    continue

                center = self.cell_center((r, c))
                x, y = float(center[0]), float(center[1])

                neighbors = [
                    (-1, 0, "horizontal", (x, y + half)),
                    (1, 0, "horizontal", (x, y - half)),
                    (0, -1, "vertical", (x - half, y)),
                    (0, 1, "vertical", (x + half, y)),
                ]

                for dr, dc, orientation, (wx, wy) in neighbors:
                    nr, nc = r + dr, c + dc
                    blocked = (
                        nr < 0 or nr >= rows
                        or nc < 0 or nc >= cols
                        or self.grid[nr][nc] == "#"
                    )
                    if not blocked:
                        continue

                    if orientation == "vertical":
                        collision_shape = vertical_collision
                        visual_shape = vertical_visual
                    else:
                        collision_shape = horizontal_collision
                        visual_shape = horizontal_visual

                    wall_id = p.createMultiBody(
                        baseMass=0,
                        baseCollisionShapeIndex=collision_shape,
                        baseVisualShapeIndex=visual_shape,
                        basePosition=[wx, wy, half_h],
                        physicsClientId=self.client,
                    )
                    self.wall_ids.append(wall_id)

    def wall_collision(self):
        for wall_id in self.wall_ids:
            contacts = p.getContactPoints(
                bodyA=self.base.drone_id,
                bodyB=wall_id,
                physicsClientId=self.client,
            )
            if contacts:
                return True

            nearby = p.getClosestPoints(
                bodyA=self.base.drone_id,
                bodyB=wall_id,
                distance=0.015,
                physicsClientId=self.client,
            )
            if nearby:
                return True
        return False

    def step(self, command):
        state, reward, done, info = self.base.step(command)
        collision = self.wall_collision()
        info["wall_collision"] = collision
        info["collision"] = bool(info.get("collision", False) or collision)
        done = bool(done or collision)
        self.state = state
        return state, reward, done, info

    def close(self):
        if not self.closed:
            self.base.close()
            self.closed = True
