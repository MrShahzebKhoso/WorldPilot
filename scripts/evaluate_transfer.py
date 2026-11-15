"""Evaluate an existing WorldPilot checkpoint on a new environment dataset."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn


class WorldModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(31, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, 28),
        )

    def forward(self, x):
        return self.net(x)


def rmse(pred, target):
    e = pred - target
    return {
        "state_rmse": float(np.sqrt(np.mean(e ** 2))),
        "position_rmse_m": float(np.sqrt(np.mean(e[..., :3] ** 2))),
        "velocity_rmse_mps": float(np.sqrt(np.mean(e[..., 3:6] ** 2))),
        "attitude_rmse_rad": float(np.sqrt(np.mean(e[..., 6:9] ** 2))),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="datasets/worldpilot_maze.npz")
    parser.add_argument("--checkpoint", default="checkpoints/worldpilot_mlp.pt")
    parser.add_argument("--output", default="outputs/maze_zero_shot_transfer.json")
    args = parser.parse_args()

    data = np.load(args.data)
    states = data["state"].astype(np.float32)
    actions = data["action"].astype(np.float32)
    next_states = data["next_state"].astype(np.float32)
    episode_ids = data["episode_id"]
    timesteps = data["timestep"]

    checkpoint = torch.load(
        args.checkpoint, map_location="cpu", weights_only=False
    )

    model = WorldModel()
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    x_mean = checkpoint["input_mean"]
    x_std = checkpoint["input_std"]
    y_mean = checkpoint["delta_mean"]
    y_std = checkpoint["delta_std"]

    x = np.concatenate([states, actions], axis=1)
    x_norm = ((x - x_mean) / x_std).astype(np.float32)

    with torch.no_grad():
        pred_delta_norm = model(torch.from_numpy(x_norm)).numpy()

    pred_next = states + pred_delta_norm * y_std + y_mean

    results = {
        "dataset": args.data,
        "checkpoint": args.checkpoint,
        "transitions": len(states),
        "episodes": int(len(np.unique(episode_ids))),
        "zero_shot_one_step_model": rmse(pred_next, next_states),
        "one_step_persistence_baseline": rmse(states, next_states),
        "multi_step_position_rollout": {},
    }

    # Closed-loop model rollout using recorded actions.
    for horizon in (5, 10, 20):
        model_positions, truth_positions = [], []
        persistence_positions = []

        for episode in np.unique(episode_ids):
            idx = np.where(episode_ids == episode)[0]
            idx = idx[np.argsort(timesteps[idx])]

            for start in range(0, len(idx) - horizon + 1, horizon):
                window = idx[start:start + horizon]
                current = states[window[0]].copy()
                initial = current.copy()

                for j in window:
                    model_input = np.concatenate([current, actions[j]])
                    normalized = ((model_input - x_mean) / x_std).astype(np.float32)

                    with torch.no_grad():
                        d_norm = model(
                            torch.from_numpy(normalized).unsqueeze(0)
                        ).numpy()[0]

                    current = current + d_norm * y_std + y_mean
                    model_positions.append(current[:3].copy())
                    truth_positions.append(next_states[j, :3].copy())
                    persistence_positions.append(initial[:3].copy())

        if model_positions:
            mp = np.asarray(model_positions)
            tp = np.asarray(truth_positions)
            pp = np.asarray(persistence_positions)

            results["multi_step_position_rollout"][str(horizon)] = {
                "model_position_rmse_m": float(np.sqrt(np.mean((mp - tp) ** 2))),
                "persistence_position_rmse_m": float(np.sqrt(np.mean((pp - tp) ** 2))),
                "samples": len(mp),
            }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, indent=2))

    print("\nZERO-SHOT INDOOR TRANSFER")
    print(json.dumps(results, indent=2))
    print("\nSaved:", output)


if __name__ == "__main__":
    main()
