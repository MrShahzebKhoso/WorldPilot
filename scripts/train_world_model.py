"""Train and evaluate a residual neural drone dynamics model."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


class WorldModel(nn.Module):
    def __init__(self, input_dim=31, output_dim=28):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim),
        )

    def forward(self, x):
        return self.net(x)


def metrics(pred, target):
    error = pred - target
    return {
        "state_rmse": float(np.sqrt(np.mean(error ** 2))),
        "position_rmse_m": float(np.sqrt(np.mean(error[..., 0:3] ** 2))),
        "velocity_rmse_mps": float(np.sqrt(np.mean(error[..., 3:6] ** 2))),
        "attitude_rmse_rad": float(np.sqrt(np.mean(error[..., 6:9] ** 2))),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="datasets/worldpilot_stage2.npz")
    parser.add_argument("--output", default="checkpoints/worldpilot_mlp.pt")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cpu")
    print("Device:", device)

    data = np.load(args.data)
    states = data["state"].astype(np.float32)
    actions = data["action"].astype(np.float32)
    next_states = data["next_state"].astype(np.float32)
    episode_ids = data["episode_id"]

    if states.ndim != 2 or states.shape[1] != 28:
        raise ValueError(f"Expected state shape (N,28), got {states.shape}")
    if actions.shape != (len(states), 3):
        raise ValueError(f"Expected action shape (N,3), got {actions.shape}")

    # Split whole episodes, not individual transitions.
    unique_episodes = np.unique(episode_ids)
    rng = np.random.default_rng(args.seed)
    rng.shuffle(unique_episodes)
    n_train = max(1, int(0.8 * len(unique_episodes)))
    train_eps = set(unique_episodes[:n_train].tolist())
    train_mask = np.isin(episode_ids, list(train_eps))
    test_mask = ~train_mask

    if not test_mask.any():
        raise ValueError("Need at least two episodes for an episode-level test split.")

    x = np.concatenate([states, actions], axis=1)
    delta = next_states - states

    x_mean = x[train_mask].mean(axis=0)
    x_std = x[train_mask].std(axis=0)
    y_mean = delta[train_mask].mean(axis=0)
    y_std = delta[train_mask].std(axis=0)

    x_std = np.maximum(x_std, 1e-6)
    y_std = np.maximum(y_std, 1e-6)

    x_norm = ((x - x_mean) / x_std).astype(np.float32)
    y_norm = ((delta - y_mean) / y_std).astype(np.float32)

    train_ds = TensorDataset(
        torch.from_numpy(x_norm[train_mask]),
        torch.from_numpy(y_norm[train_mask]),
    )
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True
    )

    model = WorldModel().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    loss_fn = nn.MSELoss()

    best_val = float("inf")
    best_state = None

    x_test = torch.from_numpy(x_norm[test_mask]).to(device)
    y_test = torch.from_numpy(y_norm[test_mask]).to(device)

    print(
        f"Transitions: {len(states)} | Episodes: {len(unique_episodes)} | "
        f"Train: {int(train_mask.sum())} | Test: {int(test_mask.sum())}"
    )

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_losses = []

        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_losses.append(loss.item())

        model.eval()
        with torch.no_grad():
            val_loss = loss_fn(model(x_test), y_test).item()

        if val_loss < best_val:
            best_val = val_loss
            best_state = {
                k: v.detach().cpu().clone()
                for k, v in model.state_dict().items()
            }

        if epoch == 1 or epoch % 10 == 0:
            print(
                f"Epoch {epoch:03d}/{args.epochs} | "
                f"train={np.mean(train_losses):.6f} | val={val_loss:.6f}"
            )

    model.load_state_dict(best_state)
    model.eval()

    with torch.no_grad():
        pred_delta_norm = model(x_test).cpu().numpy()

    pred_delta = pred_delta_norm * y_std + y_mean
    predicted_next = states[test_mask] + pred_delta
    actual_next = next_states[test_mask]

    one_step = metrics(predicted_next, actual_next)
    persistence = metrics(states[test_mask], actual_next)

    # Autoregressive rollouts: predict state, then feed it back with
    # the recorded action at the next timestep.
    rollout_errors = {}
    test_episode_ids = np.unique(episode_ids[test_mask])

    for horizon in (5, 10, 20):
        pred_positions = []
        true_positions = []

        for ep in test_episode_ids:
            indices = np.where((episode_ids == ep) & test_mask)[0]
            indices = indices[np.argsort(data["timestep"][indices])]
            if len(indices) < horizon:
                continue

            # Each rollout starts from a real state, then uses model predictions.
            for start in range(0, len(indices) - horizon + 1, horizon):
                window = indices[start:start + horizon]
                current = states[window[0]].copy()

                for idx in window:
                    model_input = np.concatenate([current, actions[idx]])
                    model_input = (model_input - x_mean) / x_std
                    tensor = torch.from_numpy(
                        model_input.astype(np.float32)
                    ).unsqueeze(0).to(device)

                    with torch.no_grad():
                        delta_norm_pred = model(tensor).cpu().numpy()[0]

                    predicted_change = delta_norm_pred * y_std + y_mean
                    current = current + predicted_change

                    pred_positions.append(current[:3].copy())
                    true_positions.append(next_states[idx, :3].copy())

        if pred_positions:
            p_arr = np.asarray(pred_positions)
            t_arr = np.asarray(true_positions)
            rollout_errors[str(horizon)] = {
                "position_rmse_m": float(
                    np.sqrt(np.mean((p_arr - t_arr) ** 2))
                ),
                "samples": len(p_arr),
            }

    results = {
        "data": args.data,
        "episodes": int(len(unique_episodes)),
        "train_transitions": int(train_mask.sum()),
        "test_transitions": int(test_mask.sum()),
        "best_validation_normalized_mse": float(best_val),
        "persistence_baseline": persistence,
        "one_step_model": one_step,
        "multi_step_position_rollout": rollout_errors,
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "model_state_dict": best_state,
        "input_mean": x_mean,
        "input_std": x_std,
        "delta_mean": y_mean,
        "delta_std": y_std,
        "state_dim": 28,
        "action_dim": 3,
        "architecture": "MLP 31-256-256-128-28 residual dynamics",
        "results": results,
    }
    torch.save(checkpoint, output_path)
    output_path.with_suffix(".json").write_text(
        json.dumps(results, indent=2), encoding="utf-8"
    )

    print("\nEVALUATION")
    print(json.dumps(results, indent=2))
    print("\nSaved model:", output_path)
    print("Saved metrics:", output_path.with_suffix(".json"))


if __name__ == "__main__":
    main()
