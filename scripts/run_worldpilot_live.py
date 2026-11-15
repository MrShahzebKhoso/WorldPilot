"""WorldPilot: live PyBullet simulation with learned dynamics predictions."""

import argparse
import csv
import time
from pathlib import Path

import cv2
import numpy as np
import pybullet as p
import torch
from torch import nn

from worldpilot.config import load_config
from worldpilot.environment import WorldPilotDroneEnv


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


class Predictor:
    def __init__(self, checkpoint_path):
        checkpoint = torch.load(
            checkpoint_path, map_location="cpu", weights_only=False
        )

        self.model = WorldModel()
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()

        self.input_mean = checkpoint["input_mean"]
        self.input_std = checkpoint["input_std"]
        self.delta_mean = checkpoint["delta_mean"]
        self.delta_std = checkpoint["delta_std"]

    def predict(self, state, action):
        x = np.concatenate([state, action]).astype(np.float32)
        x = (x - self.input_mean) / self.input_std

        with torch.no_grad():
            delta_normalized = self.model(
                torch.from_numpy(x).unsqueeze(0)
            ).numpy()[0]

        delta = delta_normalized * self.delta_std + self.delta_mean
        return (state + delta).astype(np.float32)


def predicted_rollout(model, env, initial_state, horizon):
    """Roll forward using predicted states and the waypoint controller."""
    current = initial_state.copy()
    positions = [current[:3].copy()]

    collection = env.config["collection"]

    for _ in range(horizon):
        action = env.waypoint_controller(
            current,
            exploration_sigma=0.0,
            random_action_probability=0.0,
            influence_distance=collection["obstacle_influence_distance"],
            repulsion_gain=collection["repulsion_gain"],
        )

        current = model.predict(current, action)
        positions.append(current[:3].copy())

    return np.asarray(positions)


def project_points(points, view_matrix, projection_matrix, width, height):
    """Project world coordinates into image pixel coordinates."""
    view = np.asarray(view_matrix).reshape(4, 4, order="F")
    projection = np.asarray(projection_matrix).reshape(4, 4, order="F")

    result = []

    for point in points:
        world = np.array([*point, 1.0])
        clip = projection @ view @ world

        if clip[3] <= 1e-6:
            result.append(None)
            continue

        ndc = clip[:3] / clip[3]

        if abs(ndc[0]) > 1.5 or abs(ndc[1]) > 1.5:
            result.append(None)
            continue

        px = int((ndc[0] + 1) * width / 2)
        py = int((1 - ndc[1]) * height / 2)
        result.append((px, py))

    return result


def draw_projected_path(frame, points, view, projection, color,
                        width=2, dashed=False):
    pixels = project_points(
        points, view, projection, frame.shape[1], frame.shape[0]
    )

    for i in range(len(pixels) - 1):
        a, b = pixels[i], pixels[i + 1]

        if a is None or b is None:
            continue

        if dashed and i % 2 == 1:
            continue

        cv2.line(frame, a, b, color, width, cv2.LINE_AA)


def capture_frame(env, actual_path, predicted_path, telemetry, args):
    width, height = args.width, args.height

    target = env.env.pos[0].copy()
    target[2] += 0.35

    view = p.computeViewMatrixFromYawPitchRoll(
        cameraTargetPosition=target.tolist(),
        distance=6.5,
        yaw=40,
        pitch=-27,
        roll=0,
        upAxisIndex=2,
    )

    projection = p.computeProjectionMatrixFOV(
        fov=60,
        aspect=width / height,
        nearVal=0.1,
        farVal=30,
    )

    image_data = p.getCameraImage(
        width,
        height,
        viewMatrix=view,
        projectionMatrix=projection,
        renderer=p.ER_TINY_RENDERER,
        physicsClientId=env.client,
    )

    rgba = np.asarray(image_data[2], dtype=np.uint8).reshape(
        height, width, 4
    )
    frame = cv2.cvtColor(rgba[:, :, :3], cv2.COLOR_RGB2BGR)

    # Actual path: solid blue.
    draw_projected_path(
        frame, actual_path, view, projection,
        color=(255, 110, 35), width=3
    )

    # Predicted path: dashed green.
    draw_projected_path(
        frame, predicted_path, view, projection,
        color=(70, 230, 80), width=3, dashed=True
    )

    # Telemetry overlay.
    overlay = frame.copy()
    cv2.rectangle(overlay, (15, 15), (490, 180), (15, 20, 25), -1)
    cv2.addWeighted(overlay, 0.78, frame, 0.22, 0, frame)

    lines = [
        ("WORLDPILOT | LEARNED DRONE DYNAMICS", (255, 255, 255)),
        (f"Time: {telemetry['time']:.2f} s", (230, 230, 230)),
        (f"Goal distance: {telemetry['goal_distance']:.3f} m",
         (230, 230, 230)),
        (f"One-step position error: {telemetry['error_mm']:.2f} mm",
         (70, 230, 80)),
        (f"Prediction horizon: {args.horizon} steps",
         (70, 230, 80)),
    ]

    y = 43
    for text, color in lines:
        cv2.putText(
            frame, text, (28, y),
            cv2.FONT_HERSHEY_SIMPLEX, 0.52,
            color, 1, cv2.LINE_AA
        )
        y += 29

    cv2.putText(
        frame,
        "BLUE: ACTUAL     GREEN: MODEL PREDICTION",
        (25, height - 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55, (255, 255, 255), 1, cv2.LINE_AA
    )

    return frame, view


def update_gui_lines(env, actual_path, predicted_path, gui_ids):
    """Maintain persistent actual and predicted trajectory lines."""
    if len(actual_path) >= 2:
        env.draw_trajectory_segment(
            actual_path[-2],
            actual_path[-1],
            color=(0.12, 0.45, 1.0),
            width=3,
        )

    # Green dashed predicted path.
    segments = [
        (predicted_path[i], predicted_path[i + 1])
        for i in range(0, len(predicted_path) - 1, 2)
    ]

    for i, (start, end) in enumerate(segments):
        old_id = gui_ids[i] if i < len(gui_ids) else -1

        new_id = p.addUserDebugLine(
            start.tolist(),
            end.tolist(),
            lineColorRGB=[0.1, 0.9, 0.25],
            lineWidth=2,
            lifeTime=0,
            replaceItemUniqueId=old_id,
            physicsClientId=env.client,
        )

        if i < len(gui_ids):
            gui_ids[i] = new_id
        else:
            gui_ids.append(new_id)

    # Remove obsolete prediction segments.
    while len(gui_ids) > len(segments):
        item = gui_ids.pop()
        p.removeUserDebugItem(item, physicsClientId=env.client)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--steps", type=int, default=1440)
    parser.add_argument("--horizon", type=int, default=20)
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument(
        "--checkpoint", default="checkpoints/worldpilot_mlp.pt"
    )
    parser.add_argument("--output", default="outputs")
    parser.add_argument("--config", default=None)
    args = parser.parse_args()

    torch.set_num_threads(1)

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    video_path = output_dir / "worldpilot_live.mp4"
    csv_path = output_dir / "worldpilot_live_telemetry.csv"

    print("Loading learned dynamics model...")
    model = Predictor(args.checkpoint)

    config = load_config(args.config)
    env = WorldPilotDroneEnv(config, gui=True, seed=args.seed)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(
        str(video_path), fourcc, args.fps,
        (args.width, args.height)
    )

    if not writer.isOpened():
        env.close()
        raise RuntimeError(
            "Could not initialize MP4 video writer. "
            "Check OpenCV video codec support."
        )

    capture_interval = max(
        1, round(env.ctrl_hz / args.fps)
    )

    actual_path = []
    predicted_path = []
    gui_ids = []
    latest_error_mm = 0.0

    try:
        state, _ = env.reset(seed=args.seed)
        actual_path.append(state[:3].copy())

        print("\nWorldPilot live demonstration started.")
        print("Blue = actual trajectory | Green = predicted trajectory")
        print("Press Ctrl+C to stop.\n")

        with csv_path.open("w", newline="") as file:
            log = csv.writer(file)
            log.writerow([
                "time_s",
                "x_m", "y_m", "z_m",
                "goal_distance_m",
                "one_step_position_error_mm",
                "prediction_horizon",
            ])

            start_time = time.time()

            for step in range(args.steps):
                # Current controller action.
                collection = config["collection"]
                action = env.waypoint_controller(
                    state,
                    exploration_sigma=0.0,
                    random_action_probability=0.0,
                    influence_distance=collection[
                        "obstacle_influence_distance"
                    ],
                    repulsion_gain=collection["repulsion_gain"],
                )

                # One-step prediction before advancing the real simulation.
                one_step_prediction = model.predict(state, action)

                # Advance the actual PyBullet simulation.
                next_state, _, done, info = env.step(action)

                latest_error_mm = (
                    np.linalg.norm(
                        one_step_prediction[:3] - next_state[:3]
                    ) * 1000
                )

                state = next_state
                actual_path.append(state[:3].copy())

                current_time = (step + 1) * env.dt

                log.writerow([
                    current_time,
                    *state[:3].tolist(),
                    info["goal_distance"],
                    latest_error_mm,
                    args.horizon,
                ])

                if step % capture_interval == 0 or done:
                    # Recompute a future rollout from the latest actual state.
                    predicted_path = predicted_rollout(
                        model, env, state, args.horizon
                    )

                    update_gui_lines(
                        env, actual_path, predicted_path, gui_ids
                    )

                    telemetry = {
                        "time": current_time,
                        "goal_distance": info["goal_distance"],
                        "error_mm": latest_error_mm,
                    }

                    frame, _ = capture_frame(
                        env, actual_path, predicted_path,
                        telemetry, args
                    )
                    writer.write(frame)

                if step % env.ctrl_hz == 0:
                    print(
                        f"t={current_time:5.1f}s | "
                        f"goal={info['goal_distance']:.2f}m | "
                        f"prediction error={latest_error_mm:.2f}mm"
                    )

                time.sleep(env.dt)

                if done:
                    print("\nEpisode finished:", info)
                    break

    except KeyboardInterrupt:
        print("\nDemo interrupted by user.")

    finally:
        writer.release()
        env.close()

    print("\nWorldPilot recording complete.")
    print("Video:", video_path)
    print("Telemetry:", csv_path)


if __name__ == "__main__":
    main()
