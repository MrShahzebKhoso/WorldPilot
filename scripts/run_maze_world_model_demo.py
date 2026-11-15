"""Record an indoor navigation demo with one-step-ahead world-model predictions."""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import pybullet as p
import torch
import yaml
from torch import nn

from worldpilot.maze_environment import IndoorMazeEnv
from worldpilot.maze_planner import astar
from chase_camera import render_chase_frame

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


def load_model(path):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model = WorldModel()
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, ckpt


def rounded_box(img, p1, p2, color, radius=18, thickness=-1):
    x1, y1 = p1
    x2, y2 = p2
    cv2.rectangle(img, (x1 + radius, y1), (x2 - radius, y2),
                  color, thickness)
    cv2.rectangle(img, (x1, y1 + radius), (x2, y2 - radius),
                  color, thickness)
    for x, y in ((x1 + radius, y1 + radius),
                 (x2 - radius, y1 + radius),
                 (x1 + radius, y2 - radius),
                 (x2 - radius, y2 - radius)):
        cv2.circle(img, (x, y), radius, color, thickness)


def render_frame(grid, origin_x, origin_y, cell, route,
                 actual, predicted, position, goal, step,
                 goal_distance, errors, fps):
    W, H = 1280, 800
    img = np.full((H, W, 3), (18, 22, 29), dtype=np.uint8)

    # Header
    cv2.putText(img, "WORLDPILOT", (48, 55),
                cv2.FONT_HERSHEY_SIMPLEX, 0.85, (245, 248, 252), 2,
                cv2.LINE_AA)
    cv2.putText(img, "INDOOR WORLD MODEL  /  TWO-ROOM NAVIGATION",
                (48, 91), cv2.FONT_HERSHEY_SIMPLEX, 0.48,
                (160, 177, 195), 1, cv2.LINE_AA)

    # Panels
    rounded_box(img, (35, 120), (875, 735), (29, 35, 44), 18)
    rounded_box(img, (895, 120), (1245, 735), (29, 35, 44), 18)

    cv2.putText(img, "TOP-DOWN ENVIRONMENT", (60, 155),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, (175, 190, 205), 1,
                cv2.LINE_AA)

    # Map geometry
    map_x, map_y = 65, 205
    scale = 52.0
    x_min = origin_x - cell / 2
    y_max = origin_y + cell / 2

    def xy_to_px(x, y):
        return (
            int(map_x + (x - x_min) * scale),
            int(map_y + (y_max - y) * scale),
        )

    rows, cols = len(grid), len(grid[0])
    for r in range(rows):
        for c in range(cols):
            x = origin_x + c * cell
            y = origin_y - r * cell
            px, py = xy_to_px(x, y)
            half = int(scale / 2)
            rect = (px - half, py - half, px + half, py + half)

            if grid[r][c] == "#":
                cv2.rectangle(img, (rect[0], rect[1]), (rect[2], rect[3]),
                              (72, 82, 94), -1)
                cv2.rectangle(img, (rect[0], rect[1]), (rect[2], rect[3]),
                              (88, 99, 112), 1)
            else:
                cv2.rectangle(img, (rect[0], rect[1]), (rect[2], rect[3]),
                              (43, 51, 61), -1)
                cv2.rectangle(img, (rect[0], rect[1]), (rect[2], rect[3]),
                              (54, 63, 74), 1)

    # Planned A* route
    route_px = np.array([xy_to_px(x, y) for x, y, _ in route],
                        dtype=np.int32)
    if len(route_px) > 1:
        cv2.polylines(img, [route_px], False, (220, 185, 60), 3,
                      cv2.LINE_AA)

    # Actual trajectory and model's one-step-ahead prediction sequence
    if len(actual) > 1:
        pts = np.array([xy_to_px(x, y) for x, y, _ in actual],
                       dtype=np.int32)
        cv2.polylines(img, [pts], False, (90, 215, 125), 3,
                      cv2.LINE_AA)

    if len(predicted) > 1:
        pts = np.array([xy_to_px(x, y) for x, y, _ in predicted],
                       dtype=np.int32)
        cv2.polylines(img, [pts], False, (65, 155, 245), 2,
                      cv2.LINE_AA)

    # Start, goal, drone and current prediction
    sx, sy = xy_to_px(route[0][0], route[0][1])
    gx, gy = xy_to_px(goal[0], goal[1])
    dx, dy = xy_to_px(position[0], position[1])

    cv2.circle(img, (sx, sy), 9, (230, 190, 65), -1)
    cv2.circle(img, (gx, gy), 11, (90, 220, 120), -1)
    cv2.circle(img, (dx, dy), 10, (245, 245, 245), -1)
    cv2.circle(img, (dx, dy), 15, (235, 235, 235), 2)

    cv2.putText(img, "START", (sx - 25, sy - 17),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (230, 205, 115), 1,
                cv2.LINE_AA)
    cv2.putText(img, "GOAL", (gx - 20, gy - 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (110, 230, 140), 1,
                cv2.LINE_AA)

    # Telemetry panel
    tx = 925
    cv2.putText(img, "LIVE TELEMETRY", (tx, 158),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (175, 190, 205), 1,
                cv2.LINE_AA)

    def metric(label, value, y, color=(235, 240, 245)):
        cv2.putText(img, label, (tx, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.42, (145, 162, 180), 1, cv2.LINE_AA)
        cv2.putText(img, value, (tx, y + 34), cv2.FONT_HERSHEY_SIMPLEX,
                    0.78, color, 2, cv2.LINE_AA)

    rmse_mm = 1000.0 * np.sqrt(np.mean(np.square(errors))) if errors else 0.0
    current_mm = 1000.0 * errors[-1] if errors else 0.0

    metric("SIMULATION STEP", f"{step:05d}", 220)
    metric("DISTANCE TO GOAL", f"{goal_distance:.3f} m", 315)
    metric("CURRENT PREDICTION ERROR", f"{current_mm:.2f} mm", 410,
           (90, 175, 250))
    metric("RUNNING POSITION RMSE", f"{rmse_mm:.2f} mm", 505,
           (100, 220, 145))

    # Legend
    legend_y = 665
    legend = [
        ((220, 185, 60), "A* route"),
        ((90, 215, 125), "Actual path"),
        ((65, 155, 245), "Model prediction"),
        ((245, 245, 245), "Drone"),
    ]
    for i, (color, label) in enumerate(legend):
        x = tx + (i % 2) * 155
        y = legend_y + (i // 2) * 28
        cv2.line(img, (x, y), (x + 22, y), color, 4)
        cv2.putText(img, label, (x + 30, y + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.32,
                    (190, 200, 210), 1, cv2.LINE_AA)

    cv2.putText(img, "A* PLANNING  +  LEARNED DRONE DYNAMICS",
                (48, 770), cv2.FONT_HERSHEY_SIMPLEX, 0.43,
                (145, 165, 185), 1, cv2.LINE_AA)
    return img


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint",
                        default="checkpoints/worldpilot_maze_mlp.pt")
    parser.add_argument("--output",
                        default="outputs/worldpilot_indoor_world_model.mp4")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--max-steps", type=int, default=2200)
    parser.add_argument("--frame-stride", type=int, default=2)
    parser.add_argument("--fps", type=int, default=24)
    args = parser.parse_args()

    torch.set_num_threads(1)
    model, ckpt = load_model(args.checkpoint)
    x_mean = ckpt["input_mean"]
    x_std = ckpt["input_std"]
    y_mean = ckpt["delta_mean"]
    y_std = ckpt["delta_std"]

    with open("configs/drone_config.yaml") as f:
        base_config = yaml.safe_load(f)
    with open("configs/maze_config.yaml") as f:
        maze_config = yaml.safe_load(f)

    maze = maze_config["maze"]
    env = IndoorMazeEnv(base_config, maze_config, gui=False, seed=args.seed)
    grid = env.grid
    path = astar(grid, tuple(maze["start"]), tuple(maze["goal"]))
    waypoints = [env.cell_center(cell) for cell in path]

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output), cv2.VideoWriter_fourcc(*"mp4v"),
        args.fps, (1280, 800)
    )
    if not writer.isOpened():
        env.close()
        raise RuntimeError(f"Could not open video writer: {output}")

    actual_path = [env.state[:3].copy()]
    predicted_path = []
    position_errors = []
    waypoint_index = 1
    success = False
    collision = False
    out_of_bounds = False
    final_step = 0

    print("Recording model-prediction demo...")
    print("A* waypoints:", len(path))

    camera_state = {"eye": None, "target": None, "heading": np.array([1.0, 0.0])}

    try:
        for step in range(args.max_steps):
            state = env.state.copy()
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
                -0.12, 0.12
            )
            action = np.array(
                [horizontal[0], horizontal[1], vertical],
                dtype=np.float32
            )

            # Predict next state before advancing the simulator.
            model_input = np.concatenate([state, action])
            model_input = ((model_input - x_mean) / x_std).astype(np.float32)

            with torch.no_grad():
                delta_norm = model(
                    torch.from_numpy(model_input).unsqueeze(0)
                ).numpy()[0]

            predicted_next = state + delta_norm * y_std + y_mean

            next_state, _, done, info = env.step(action)
            actual_next = next_state.copy()

            prediction_error = float(
                np.linalg.norm(predicted_next[:3] - actual_next[:3])
            )
            position_errors.append(prediction_error)
            predicted_path.append(predicted_next[:3].copy())
            actual_path.append(actual_next[:3].copy())

            final_step = step + 1
            success |= bool(info.get("goal_reached", False))
            collision |= bool(info.get("collision", False))
            out_of_bounds |= bool(info.get("out_of_bounds", False))

            if step % args.frame_stride == 0 or done:

                # Forecast the next 20 simulation steps from the actual next state.
                future_predictions = rollout_future(
                    model,
                    actual_next,
                    waypoints,
                    waypoint_index,
                    maze,
                    x_mean,
                    x_std,
                    y_mean,
                    y_std,
                    horizon=20,
                )

                frame, camera_state = render_chase_frame(
                    env,
                    actual_path,
                    future_predictions,
                    waypoints,
                    action,
                    final_step,
                    info["goal_distance"],
                    prediction_error,
                    success,
                    collision,
                    out_of_bounds,
                    camera_state,
                )

                writer.write(frame)

            if step % 200 == 0:
                print(
                    f"step={step:4d} | goal_dist={info['goal_distance']:.2f}m "
                    f"| pred_error={prediction_error * 1000:.2f}mm"
                )

            if done:
                break

    finally:
        writer.release()
        env.close()

    results = {
        "checkpoint": args.checkpoint,
        "dataset": "datasets/worldpilot_maze.npz",
        "success": success,
        "collision": collision,
        "out_of_bounds": out_of_bounds,
        "steps": final_step,
        "waypoints": len(path),
        "prediction_type": "one-step-ahead, conditioned on actual state and commanded action",
        "position_prediction_rmse_m": float(
            np.sqrt(np.mean(np.square(position_errors)))
        ) if position_errors else None,
        "position_prediction_mean_error_m": float(
            np.mean(position_errors)
        ) if position_errors else None,
        "final_position": env.state[:3].tolist(),
        "goal_position": env.base.goal.tolist(),
        "video": str(output),
    }

    metrics_path = output.with_suffix(".json")
    metrics_path.write_text(json.dumps(results, indent=2))
    print("\nDEMO RESULT")
    print(json.dumps(results, indent=2))
    print("\nVideo:", output)
    print("Metrics:", metrics_path)


def rollout_future(
    model, initial_state, waypoints, initial_waypoint_index,
    maze, x_mean, x_std, y_mean, y_std, horizon=20
):
    """Autoregressive world-model rollout using the waypoint controller."""

    state = initial_state.copy()
    waypoint_index = initial_waypoint_index
    predictions = []

    for _ in range(horizon):
        position = state[:3].copy()

        target = waypoints[
            min(waypoint_index, len(waypoints) - 1)
        ]

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
            -0.12, 0.12
        )

        action = np.array(
            [horizontal[0], horizontal[1], vertical],
            dtype=np.float32
        )

        model_input = np.concatenate([state, action])
        model_input = (
            (model_input - x_mean) / x_std
        ).astype(np.float32)

        with torch.no_grad():
            delta_norm = model(
                torch.from_numpy(model_input).unsqueeze(0)
            ).numpy()[0]

        predicted_delta = delta_norm * y_std + y_mean
        state = state + predicted_delta

        predictions.append(state[:3].copy())

    return predictions


if __name__ == "__main__":
    main()
