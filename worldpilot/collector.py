"""Collect labeled drone transition data for WorldPilot."""
from pathlib import Path
import json
from datetime import datetime, timezone

import numpy as np
from worldpilot.environment import WorldPilotDroneEnv


def collect_dataset(
    config,
    episodes=40,
    max_steps=576,
    output="datasets/worldpilot_stage2.npz",
    gui=False,
    seed=7,
    render_delay=False,
):
    env = WorldPilotDroneEnv(config, gui=gui, seed=seed)

    fields = [
        "state", "action", "next_state",
        "collision", "goal_reached", "out_of_bounds", "done",
        "episode_id", "timestep", "trajectory_type",
    ]
    rows = {key: [] for key in fields}
    summaries = []
    collection = config["collection"]

    try:
        for episode in range(int(episodes)):
            state, _ = env.reset(seed=seed + episode)
            success = collision = out_of_bounds = False
            steps = 0

            for t in range(int(max_steps)):
                action = env.waypoint_controller(
                    state,
                    exploration_sigma=float(collection["exploration_sigma"]),
                    random_action_probability=float(
                        collection["random_action_probability"]
                    ),
                    random_action_scale=float(collection["random_action_scale"]),
                    influence_distance=float(
                        collection["obstacle_influence_distance"]
                    ),
                    repulsion_gain=float(collection["repulsion_gain"]),
                )

                next_state, _, done, info = env.step(action)

                values = {
                    "state": state.copy(),
                    "action": action.copy(),
                    "next_state": next_state.copy(),
                    "collision": bool(info["collision"]),
                    "goal_reached": bool(info["goal_reached"]),
                    "out_of_bounds": bool(info["out_of_bounds"]),
                    "done": bool(done),
                    "episode_id": episode,
                    "timestep": t,
                    "trajectory_type": "waypoint_exploratory",
                }
                for key, value in values.items():
                    rows[key].append(value)

                if gui:
                    env.draw_trajectory_segment(
                        state[:3], next_state[:3],
                        color=(0.15, 0.45, 1.0), lifetime=0.0
                    )
                    if render_delay:
                        import time
                        time.sleep(env.dt)

                state = next_state
                steps = t + 1
                success = success or info["goal_reached"]
                collision = collision or info["collision"]
                out_of_bounds = out_of_bounds or info["out_of_bounds"]

                if done:
                    break

            distance = float(np.linalg.norm(state[9:12]))
            summaries.append({
                "episode": episode,
                "steps": steps,
                "goal_reached": bool(success),
                "collision": bool(collision),
                "out_of_bounds": bool(out_of_bounds),
                "final_goal_distance_m": distance,
            })
            print(
                f"Episode {episode + 1:03d}/{episodes} | "
                f"steps={steps:3d} | goal={success} | "
                f"collision={collision} | OOB={out_of_bounds} | "
                f"goal distance={distance:.2f}m"
            )
    finally:
        env.close()

    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

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
        raise RuntimeError("No transitions collected; dataset was not saved.")

    np.savez_compressed(output_path, **arrays)

    metadata = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "project": "WorldPilot",
        "stage": "stage2_exploratory_transition_collection",
        "episodes_requested": int(episodes),
        "transitions": int(len(arrays["state"])),
        "state_dim": int(arrays["state"].shape[1]),
        "action_dim": int(arrays["action"].shape[1]),
        "seed": int(seed),
        "episode_summaries": summaries,
    }
    output_path.with_suffix(".json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )

    print(f"\nSaved {len(arrays['state'])} transitions: {output_path}")
    print(f"Metadata: {output_path.with_suffix('.json')}")
    print(
        "Events | collisions:",
        int(arrays["collision"].sum()),
        "| out of bounds:",
        int(arrays["out_of_bounds"].sum()),
        "| goal flags:",
        int(arrays["goal_reached"].sum()),
    )
    return output_path
