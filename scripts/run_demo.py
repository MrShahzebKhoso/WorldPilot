"""Run one live waypoint-navigation episode in the PyBullet GUI."""
import argparse
import time
import numpy as np
from worldpilot.config import load_config
from worldpilot.environment import WorldPilotDroneEnv


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--steps", type=int, default=900)
    parser.add_argument("--config", default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    env = WorldPilotDroneEnv(config, gui=True, seed=args.seed)
    try:
        state, _ = env.reset(seed=args.seed)
        previous = state[:3].copy()
        for step in range(args.steps):
            action = env.waypoint_controller(
                state,
                exploration_sigma=0.0,
                random_action_probability=0.0,
                influence_distance=config["collection"]["obstacle_influence_distance"],
                repulsion_gain=config["collection"]["repulsion_gain"],
            )
            next_state, _, done, info = env.step(action)
            env.draw_trajectory_segment(previous, next_state[:3], color=(0.12, 0.45, 1.0))
            previous = next_state[:3].copy()
            state = next_state
            if step % 48 == 0:
                print(f"t={step / env.ctrl_hz:5.1f}s | goal distance={info['goal_distance']:.2f}m")
            time.sleep(env.dt)
            if done:
                print("Episode finished:", info)
                break
    except KeyboardInterrupt:
        print("Demo interrupted.")
    finally:
        env.close()


if __name__ == "__main__":
    main()
