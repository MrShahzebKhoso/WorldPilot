import argparse
import yaml
import numpy as np
import pybullet as p

from worldpilot.maze_environment import IndoorMazeEnv
from worldpilot.maze_planner import astar


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    with open("configs/drone_config.yaml") as f:
        base_config = yaml.safe_load(f)

    with open("configs/maze_config.yaml") as f:
        maze_config = yaml.safe_load(f)

    env = IndoorMazeEnv(
        base_config,
        maze_config,
        gui=not args.headless,
        seed=args.seed,
    )

    try:
        grid = env.grid
        start = env.start
        goal = env.goal_cell
        path = astar(grid, start, goal)
        waypoints = [env.cell_center(cell) for cell in path]

        print("\nMAZE ROUTE")
        print("Grid cells:", len(path))
        print("Path:", path)

        if not args.headless:
            p.resetDebugVisualizerCamera(
                cameraDistance=13,
                cameraYaw=0,
                cameraPitch=-89,
                cameraTargetPosition=[0, 0, 1.0],
                physicsClientId=env.client,
            )

            for a, b in zip(waypoints[:-1], waypoints[1:]):
                p.addUserDebugLine(
                    a.tolist(), b.tolist(),
                    [0.1, 0.8, 1.0], 3,
                    physicsClientId=env.client,
                )

        waypoint_index = 1
        previous_position = env.state[:3].copy()
        actual_path = [previous_position.copy()]
        max_steps = int(maze_config["maze"]["max_steps"])
        speed_limit = float(maze_config["maze"]["speed"])
        tolerance = float(maze_config["maze"]["waypoint_tolerance"])
        altitude = float(maze_config["maze"]["flight_altitude"])

        success = False

        for step in range(max_steps):
            state = env.state
            position = state[:3].copy()

            target = waypoints[min(waypoint_index, len(waypoints) - 1)]
            delta = target - position
            distance = float(np.linalg.norm(delta[:2]))

            if distance < tolerance:
                if waypoint_index < len(waypoints) - 1:
                    waypoint_index += 1
                    target = waypoints[waypoint_index]
                    delta = target - position
                    distance = float(np.linalg.norm(delta[:2]))

            # Conservative waypoint-following velocity controller.
            if distance > 1e-6:
                speed = min(speed_limit, 0.9 * distance)
                horizontal = delta[:2] / distance * speed
            else:
                horizontal = np.zeros(2, dtype=np.float32)

            vertical = np.clip((altitude - position[2]) * 0.8, -0.12, 0.12)
            command = np.array(
                [horizontal[0], horizontal[1], vertical],
                dtype=np.float32,
            )

            next_state, _, done, info = env.step(command)
            new_position = next_state[:3].copy()
            actual_path.append(new_position)

            if not args.headless and step % 3 == 0:
                p.addUserDebugLine(
                    previous_position.tolist(),
                    new_position.tolist(),
                    [1.0, 0.25, 0.1],
                    3,
                    lifeTime=0,
                    physicsClientId=env.client,
                )

            previous_position = new_position

            if step % 100 == 0:
                print(
                    f"step={step:4d} "
                    f"position={np.round(new_position, 2)} "
                    f"goal_distance={info['goal_distance']:.2f}m"
                )

            if info.get("collision", False):
                print("STOP: collision detected", info)
                break

            if info.get("out_of_bounds", False):
                print("STOP: drone left the permitted bounds")
                break

            if info.get("goal_reached", False):
                success = True
                print(f"GOAL REACHED at step {step}")
                break

            if done:
                break

        print("\nMAZE RESULT")
        print("Success:", success)
        print("Collision:", env.wall_collision())
        print("Waypoints:", len(path))
        print("Actual trajectory samples:", len(actual_path))
        print("Final position:", np.round(env.state[:3], 3))
        print("Goal:", np.round(env.base.goal, 3))

        if not args.headless:
            print("\nClose the PyBullet window to finish.")
            while p.isConnected(physicsClientId=env.client):
                p.configureDebugVisualizer(
                    p.COV_ENABLE_SINGLE_STEP_RENDERING,
                    1,
                    physicsClientId=env.client,
                )
                import time
                time.sleep(0.05)

    finally:
        env.close()


if __name__ == "__main__":
    main()
