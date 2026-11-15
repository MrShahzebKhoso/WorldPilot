"""Collect indoor drone transitions in the two-room maze."""
import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import yaml

from worldpilot.maze_environment import IndoorMazeEnv
from worldpilot.maze_planner import astar


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--max-steps", type=int, default=2200)
    parser.add_argument("--output", default="datasets/worldpilot_maze.npz")
    parser.add_argument("--seed", type=int, default=100)
    args = parser.parse_args()

    with open("configs/drone_config.yaml") as f:
        base_config = yaml.safe_load(f)
    with open("configs/maze_config.yaml") as f:
        maze_config = yaml.safe_load(f)

    maze = maze_config["maze"]
    path = astar(
        maze["rows"], tuple(maze["start"]), tuple(maze["goal"])
    )

    fields = [
        "state", "action", "next_state",
        "collision", "goal_reached", "out_of_bounds", "done",
        "episode_id", "timestep", "trajectory_type",
    ]
    rows = {key: [] for key in fields}
    summaries = []

    print("Indoor route:", path)
    print("Waypoints:", len(path))

    for episode in range(args.episodes):
        seed = args.seed + episode
        rng = np.random.default_rng(seed)

        # A fresh simulation ensures the maze walls are present every episode.
        env = IndoorMazeEnv(
            base_config, maze_config, gui=False, seed=seed
        )

        try:
            waypoints = [env.cell_center(cell) for cell in path]
            state = env.state.copy()
            waypoint_index = 1

            # Vary perturbation strength across episodes.
            noise_sigma = float(rng.uniform(0.012, 0.055))
            success = collision = out_of_bounds = False
            steps = 0

            for t in range(args.max_steps):
                position = state[:3].copy()
                target = waypoints[min(waypoint_index, len(waypoints) - 1)]
                delta = target - position
                distance = float(np.linalg.norm(delta[:2]))

                if distance < float(maze["waypoint_tolerance"]):
                    if waypoint_index < len(waypoints) - 1:
                        waypoint_index += 1
                        target = waypoints[waypoint_index]
                        delta = target - position
                        distance = float(np.linalg.norm(delta[:2]))

                if distance > 1e-6:
                    speed = min(float(maze["speed"]), 0.9 * distance)
                    horizontal = delta[:2] / distance * speed
                else:
                    horizontal = np.zeros(2, dtype=np.float32)

                vertical = np.clip(
                    (float(maze["flight_altitude"]) - position[2]) * 0.8,
                    -0.12, 0.12,
                )

                action = np.array(
                    [horizontal[0], horizontal[1], vertical],
                    dtype=np.float32,
                )

                # Small, bounded command perturbations create varied dynamics.
                action += rng.normal(0.0, noise_sigma, size=3).astype(np.float32)
                if rng.random() < 0.025:
                    action[:2] += rng.normal(0.0, 0.08, size=2).astype(np.float32)

                action_norm = float(np.linalg.norm(action))
                max_speed = float(env.base.max_speed)
                if action_norm > max_speed:
                    action *= max_speed / action_norm

                next_state, _, done, info = env.step(action)

                values = {
                    "state": state.copy(),
                    "action": action.copy(),
                    "next_state": next_state.copy(),
                    "collision": bool(info.get("collision", False)),
                    "goal_reached": bool(info.get("goal_reached", False)),
                    "out_of_bounds": bool(info.get("out_of_bounds", False)),
                    "done": bool(done),
                    "episode_id": episode,
                    "timestep": t,
                    "trajectory_type": "maze_astar_noisy",
                }
                for key, value in values.items():
                    rows[key].append(value)

                state = next_state.copy()
                steps = t + 1
                success |= bool(info.get("goal_reached", False))
                collision |= bool(info.get("collision", False))
                out_of_bounds |= bool(info.get("out_of_bounds", False))

                if done:
                    break

            summary = {
                "episode": episode,
                "seed": seed,
                "steps": steps,
                "noise_sigma": noise_sigma,
                "goal_reached": success,
                "collision": collision,
                "out_of_bounds": out_of_bounds,
                "final_goal_distance_m": float(np.linalg.norm(state[9:12])),
            }
            summaries.append(summary)

            print(
                f"Episode {episode + 1:02d}/{args.episodes} | "
                f"steps={steps:4d} | success={success} | "
                f"collision={collision} | OOB={out_of_bounds} | "
                f"goal_dist={summary['final_goal_distance_m']:.2f}m"
            )

        finally:
            env.close()

    arrays = {
        "state": np.asarray(rows["state"], dtype=np.float32),
        "action": np.asarray(rows["action"], dtype=np.float32),
        "next_state": np.asarray(rows["next_state"], dtype=np.float32),
        "collision": np.asarray(rows["collision"], dtype=np.bool_),
        "goal_reached": np.asarray(rows["goal_reached"], dtype=np.bool_),
        "out_of_bounds": np.asarray(rows["out_of_bounds"], dtype=np.bool_),
        "done": np.asarray(rows["done"], dtype=np.bool_),
        "episode_id": np.asarray(rows["episode_id"], dtype=np.int32),
        "timestep": np.asarray(rows["timestep"], dtype=np.int32),
        "trajectory_type": np.asarray(rows["trajectory_type"], dtype="U32"),
    }

    if len(arrays["state"]) == 0:
        raise RuntimeError("No transitions collected.")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **arrays)

    metadata = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "project": "WorldPilot",
        "stage": "indoor_maze_transition_collection",
        "episodes_requested": args.episodes,
        "transitions": len(arrays["state"]),
        "state_dim": arrays["state"].shape[1],
        "action_dim": arrays["action"].shape[1],
        "route": [list(cell) for cell in path],
        "seed": args.seed,
        "episode_summaries": summaries,
    }
    output.with_suffix(".json").write_text(json.dumps(metadata, indent=2))

    print(f"\nSaved: {output}")
    print("Transitions:", len(arrays["state"]))
    print("State/action:", arrays["state"].shape, arrays["action"].shape)
    print("Goal flags:", int(arrays["goal_reached"].sum()))
    print("Collision flags:", int(arrays["collision"].sum()))
    print("Out-of-bounds flags:", int(arrays["out_of_bounds"].sum()))


if __name__ == "__main__":
    main()
