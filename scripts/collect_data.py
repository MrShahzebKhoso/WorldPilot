import argparse
from worldpilot.config import load_config
from worldpilot.collector import collect_dataset


def main():
    parser = argparse.ArgumentParser(description="Collect WorldPilot drone transitions.")
    parser.add_argument("--episodes", type=int, default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--output", default="datasets/worldpilot_stage1.npz")
    parser.add_argument("--gui", action="store_true", help="Show the PyBullet GUI.")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--realtime", action="store_true", help="Slow GUI rollout to real time.")
    parser.add_argument("--config", default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    episodes = args.episodes or config["collection"]["default_episodes"]
    max_steps = args.max_steps or config["collection"]["max_steps_per_episode"]
    seed = args.seed if args.seed is not None else config["collection"]["seed"]

    collect_dataset(
        config, episodes=episodes, max_steps=max_steps,
        output=args.output, gui=args.gui, seed=seed, render_delay=args.realtime,
    )


if __name__ == "__main__":
    main()
