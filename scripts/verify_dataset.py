"""Validate the Stage 1 NPZ dataset and print useful statistics."""
import argparse
from pathlib import Path
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="datasets/worldpilot_stage1.npz")
    args = parser.parse_args()
    path = Path(args.dataset)
    if not path.exists():
        raise SystemExit(f"Dataset not found: {path}. Run scripts/collect_data.py first.")

    data = np.load(path)
    required = ["state", "action", "next_state", "collision", "goal_reached",
                "episode_id", "timestep"]
    missing = [key for key in required if key not in data]
    if missing:
        raise SystemExit(f"Missing arrays: {missing}")

    s, a, ns = data["state"], data["action"], data["next_state"]
    n = len(s)
    assert s.shape == ns.shape == (n, 28), (s.shape, ns.shape)
    assert a.shape == (n, 3), a.shape
    assert np.isfinite(s).all() and np.isfinite(a).all() and np.isfinite(ns).all()
    assert len(data["collision"]) == n and len(data["goal_reached"]) == n

    delta = ns[:, :3] - s[:, :3]
    print("WORLD PILOT STAGE 1 DATASET: PASS")
    print(f"Transitions:          {n:,}")
    print(f"Episodes:              {len(np.unique(data['episode_id']))}")
    print(f"State shape:           {s.shape}")
    print(f"Action shape:          {a.shape}")
    print(f"Position delta mean:   {np.linalg.norm(delta, axis=1).mean():.4f} m/step")
    print(f"Max action magnitude:  {np.linalg.norm(a, axis=1).max():.4f} m/s")
    print(f"Collision transitions:{int(data['collision'].sum()):6d}")
    print(f"Goal reached flags:    {int(data['goal_reached'].sum()):6d}")
    print(f"Finite values:         PASS")


if __name__ == "__main__":
    main()
