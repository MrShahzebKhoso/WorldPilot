"""Evaluate and visualize WorldPilot on held-out episodes."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn


class WorldModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(31, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 28),
        )

    def forward(self, x):
        return self.net(x)


def predict_next(model, state, action, checkpoint):
    """One residual-dynamics prediction."""
    model_input = np.concatenate([state, action]).astype(np.float32)
    x_mean = checkpoint["input_mean"]
    x_std = checkpoint["input_std"]
    y_mean = checkpoint["delta_mean"]
    y_std = checkpoint["delta_std"]

    normalized = (model_input - x_mean) / x_std
    tensor = torch.from_numpy(normalized).unsqueeze(0)

    with torch.no_grad():
        delta_normalized = model(tensor).numpy()[0]

    delta = delta_normalized * y_std + y_mean
    return state + delta


def contiguous_groups(indices, episode_ids, timesteps):
    """Return sorted, contiguous test sequences, grouped by episode."""
    groups = []

    for episode in np.unique(episode_ids[indices]):
        selected = indices[episode_ids[indices] == episode]
        selected = selected[np.argsort(timesteps[selected])]

        if len(selected) == 0:
            continue

        current = [selected[0]]

        for idx in selected[1:]:
            previous = current[-1]
            if timesteps[idx] == timesteps[previous] + 1:
                current.append(idx)
            else:
                if len(current) >= 2:
                    groups.append(np.asarray(current))
                current = [idx]

        if len(current) >= 2:
            groups.append(np.asarray(current))

    return groups


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data", default="datasets/worldpilot_stage2.npz"
    )
    parser.add_argument(
        "--checkpoint", default="checkpoints/worldpilot_mlp.pt"
    )
    parser.add_argument("--output", default="outputs")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--plot-horizon", type=int, default=20)
    args = parser.parse_args()

    torch.set_num_threads(1)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading checkpoint...")
    checkpoint = torch.load(
        args.checkpoint, map_location="cpu", weights_only=False
    )

    model = WorldModel()
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    data = np.load(args.data)
    states = data["state"].astype(np.float32)
    actions = data["action"].astype(np.float32)
    next_states = data["next_state"].astype(np.float32)
    episode_ids = data["episode_id"]
    timesteps = data["timestep"]

    # Reproduce the episode-level split used during training.
    unique_episodes = np.unique(episode_ids)
    rng = np.random.default_rng(args.seed)
    shuffled = unique_episodes.copy()
    rng.shuffle(shuffled)
    n_train = max(1, int(0.8 * len(unique_episodes)))
    train_episodes = set(shuffled[:n_train].tolist())
    test_mask = ~np.isin(episode_ids, list(train_episodes))
    test_indices = np.where(test_mask)[0]

    print(
        f"Episodes: {len(unique_episodes)} | "
        f"Held-out episodes: {len(unique_episodes) - n_train} | "
        f"Test transitions: {len(test_indices)}"
    )

    # One-step predictions on every held-out transition.
    predictions = np.stack([
        predict_next(model, states[i], actions[i], checkpoint)
        for i in test_indices
    ])
    targets = next_states[test_indices]
    errors = predictions - targets

    position_rmse = float(np.sqrt(np.mean(errors[:, :3] ** 2)))
    velocity_rmse = float(np.sqrt(np.mean(errors[:, 3:6] ** 2)))
    attitude_rmse = float(np.sqrt(np.mean(errors[:, 6:9] ** 2)))

    print("\nHELD-OUT ONE-STEP RESULTS")
    print(f"Position RMSE: {position_rmse:.6f} m")
    print(f"Velocity RMSE: {velocity_rmse:.6f} m/s")
    print(f"Attitude RMSE: {attitude_rmse:.6f} rad")

    # Build contiguous held-out sequences for rollout evaluation.
    groups = contiguous_groups(test_indices, episode_ids, timesteps)
    max_horizon = min(30, max(len(g) - 1 for g in groups))
    rollout_errors = {h: [] for h in range(1, max_horizon + 1)}

    for group in groups:
        # Use multiple starting points, without crossing episode boundaries.
        for start in range(0, len(group) - 1, 5):
            available = min(max_horizon, len(group) - start - 1)
            if available < 1:
                continue

            current = states[group[start]].copy()

            for step in range(available):
                idx = group[start + step]
                current = predict_next(
                    model, current, actions[idx], checkpoint
                )
                true_position = next_states[idx, :3]
                error = np.linalg.norm(current[:3] - true_position)
                rollout_errors[step + 1].append(error)

    horizons = []
    rollout_rmse = []

    for horizon, values in rollout_errors.items():
        if values:
            horizons.append(horizon)
            rollout_rmse.append(
                float(np.sqrt(np.mean(np.square(values))))
            )

    # Select the longest held-out episode for a representative plot.
    longest = max(groups, key=len)
    plot_horizon = min(args.plot_horizon, len(longest) - 1)
    plot_indices = longest[:plot_horizon]

    actual_positions = [states[plot_indices[0], :3].copy()]
    predicted_positions = [states[plot_indices[0], :3].copy()]
    current = states[plot_indices[0]].copy()

    for idx in plot_indices:
        current = predict_next(model, current, actions[idx], checkpoint)
        predicted_positions.append(current[:3].copy())
        actual_positions.append(next_states[idx, :3].copy())

    actual_positions = np.asarray(actual_positions)
    predicted_positions = np.asarray(predicted_positions)
    time_axis = np.arange(len(actual_positions)) / 48.0

    # 1. 3D trajectory comparison.
    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(
        actual_positions[:, 0],
        actual_positions[:, 1],
        actual_positions[:, 2],
        label="Actual",
        linewidth=2.5,
    )
    ax.plot(
        predicted_positions[:, 0],
        predicted_positions[:, 1],
        predicted_positions[:, 2],
        "--",
        label="World model prediction",
        linewidth=2,
    )
    ax.scatter(*actual_positions[0], marker="o", s=55, label="Start")
    ax.scatter(*actual_positions[-1], marker="x", s=65, label="End")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.set_title(f"Actual vs Predicted Drone Trajectory ({plot_horizon} steps)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "trajectory_3d.png", dpi=180)
    plt.close(fig)

    # 2. Position components over time.
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    labels = ["X", "Y", "Z"]

    for dim, ax in enumerate(axes):
        ax.plot(time_axis, actual_positions[:, dim], label="Actual")
        ax.plot(
            time_axis,
            predicted_positions[:, dim],
            "--",
            label="Predicted",
        )
        ax.set_ylabel(f"{labels[dim]} (m)")
        ax.grid(True, alpha=0.3)
        ax.legend()

    axes[-1].set_xlabel("Time (s)")
    fig.suptitle("Drone Position: Actual vs World Model")
    fig.tight_layout()
    fig.savefig(output_dir / "position_tracking.png", dpi=180)
    plt.close(fig)

    # 3. Rollout RMSE as prediction horizon increases.
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(horizons, np.asarray(rollout_rmse) * 1000, marker="o")
    ax.set_xlabel("Prediction horizon (control steps)")
    ax.set_ylabel("Position RMSE (mm)")
    ax.set_title("Open-loop Rollout Error")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "rollout_error.png", dpi=180)
    plt.close(fig)

    # 4. One-step error distributions.
    position_error = np.linalg.norm(errors[:, :3], axis=1)
    velocity_error = np.linalg.norm(errors[:, 3:6], axis=1)
    attitude_error = np.linalg.norm(errors[:, 6:9], axis=1)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    distributions = [
        (position_error * 1000, "Position error (mm)"),
        (velocity_error, "Velocity error (m/s)"),
        (attitude_error, "Attitude error (rad)"),
    ]

    for ax, (values, label) in zip(axes, distributions):
        ax.hist(values, bins=45)
        ax.set_xlabel(label)
        ax.set_ylabel("Count")
        ax.grid(True, alpha=0.25)

    fig.suptitle("Held-out One-step Prediction Error")
    fig.tight_layout()
    fig.savefig(output_dir / "error_distributions.png", dpi=180)
    plt.close(fig)

    summary = {
        "held_out_episodes": int(len(unique_episodes) - n_train),
        "test_transitions": int(len(test_indices)),
        "one_step_position_rmse_m": position_rmse,
        "one_step_velocity_rmse_mps": velocity_rmse,
        "one_step_attitude_rmse_rad": attitude_rmse,
        "rollout_position_rmse_m": {
            str(h): float(v)
            for h, v in zip(horizons, rollout_rmse)
        },
        "representative_episode": int(episode_ids[longest[0]]),
        "representative_plot_steps": int(plot_horizon),
        "note": (
            "Rollouts use recorded actions and predicted states; "
            "they are open-loop model evaluations, not closed-loop control."
        ),
    }

    (output_dir / "visualization_metrics.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    print("\nSaved visualizations:")
    for filename in [
        "trajectory_3d.png",
        "position_tracking.png",
        "rollout_error.png",
        "error_distributions.png",
        "visualization_metrics.json",
    ]:
        print(f"  {output_dir / filename}")

    print("\nRollout RMSE:")
    for h, value in zip(horizons, rollout_rmse):
        print(f"  {h:2d} steps: {value:.6f} m")


if __name__ == "__main__":
    main()
