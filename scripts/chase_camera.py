
import cv2
import numpy as np
import pybullet as p


WIDTH, HEIGHT = 1280, 800


# BGR colors
COLOR_ASTAR = (255, 190, 40)       # Cyan/blue
COLOR_ACTUAL = (50, 220, 70)       # Green
COLOR_PREDICTED = (0, 145, 255)    # Orange
COLOR_WHITE = (245, 245, 245)
COLOR_PANEL = (12, 18, 25)


def _project(points, view, projection):
    """Project 3D world coordinates into camera-image pixels."""
    if not points:
        return []

    V = np.asarray(view, dtype=np.float64).reshape(
        4, 4, order="F"
    )
    P = np.asarray(projection, dtype=np.float64).reshape(
        4, 4, order="F"
    )

    pixels = []

    for point in points:
        world = np.array([
            point[0], point[1], point[2], 1.0
        ])

        clip = P @ V @ world

        if clip[3] <= 1e-6:
            pixels.append(None)
            continue

        ndc = clip[:3] / clip[3]

        x = int((ndc[0] + 1) * WIDTH / 2)
        y = int((1 - ndc[1]) * HEIGHT / 2)

        if (
            -100 <= x <= WIDTH + 100
            and -100 <= y <= HEIGHT + 100
        ):
            pixels.append((x, y))
        else:
            pixels.append(None)

    return pixels


def _draw_world_path(
    frame, points, view, projection,
    color, thickness=3, markers=False
):
    """Draw a solid projected 3D path."""

    pixels = _project(points, view, projection)

    for a, b in zip(pixels[:-1], pixels[1:]):
        if a is not None and b is not None:
            cv2.line(
                frame, a, b, color,
                thickness, cv2.LINE_AA
            )

    if markers:
        for point in pixels:
            if point is not None:
                cv2.circle(
                    frame, point, 3,
                    color, -1, cv2.LINE_AA
                )

    if pixels and pixels[-1] is not None:
        cv2.circle(
            frame, pixels[-1], 5,
            color, -1, cv2.LINE_AA
        )


def _draw_dashed_world_path(
    frame, points, view, projection,
    color, dash_length=9, gap_length=7, thickness=3
):
    """Draw a projected 3D path as a dashed line."""

    pixels = _project(points, view, projection)

    for a, b in zip(pixels[:-1], pixels[1:]):
        if a is None or b is None:
            continue

        a = np.asarray(a, dtype=np.float64)
        b = np.asarray(b, dtype=np.float64)

        segment = b - a
        length = np.linalg.norm(segment)

        if length < 1:
            continue

        direction = segment / length
        offset = 0.0

        while offset < length:
            end = min(offset + dash_length, length)

            p1 = tuple(
                np.round(a + direction * offset).astype(int)
            )
            p2 = tuple(
                np.round(a + direction * end).astype(int)
            )

            cv2.line(
                frame, p1, p2,
                color, thickness, cv2.LINE_AA
            )

            offset += dash_length + gap_length

    if pixels and pixels[-1] is not None:
        cv2.circle(
            frame, pixels[-1], 5,
            color, -1, cv2.LINE_AA
        )


def _draw_legend(frame):
    """Draw a clean, persistent trajectory legend."""

    x0 = 28
    y0 = HEIGHT - 112

    panel_width = 445
    panel_height = 94

    overlay = frame.copy()

    cv2.rectangle(
        overlay,
        (x0, y0),
        (x0 + panel_width, y0 + panel_height),
        COLOR_PANEL,
        -1,
    )

    cv2.addWeighted(
        overlay, 0.84,
        frame, 0.16,
        0, frame
    )

    # Panel border
    cv2.rectangle(
        frame,
        (x0, y0),
        (x0 + panel_width, y0 + panel_height),
        (90, 100, 110),
        1,
        cv2.LINE_AA,
    )

    cv2.putText(
        frame,
        "TRAJECTORY LEGEND",
        (x0 + 15, y0 + 21),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.43,
        (175, 190, 205),
        1,
        cv2.LINE_AA,
    )

    entries = [
        ("A* global route", COLOR_ASTAR, "solid"),
        ("Actual drone trajectory", COLOR_ACTUAL, "solid"),
        ("World-model 20-step forecast", COLOR_PREDICTED, "dashed"),
    ]

    for i, (label, color, style) in enumerate(entries):

        y = y0 + 43 + i * 23

        if style == "solid":
            cv2.line(
                frame,
                (x0 + 17, y),
                (x0 + 57, y),
                color,
                3,
                cv2.LINE_AA,
            )
        else:
            for start in range(x0 + 17, x0 + 58, 12):
                cv2.line(
                    frame,
                    (start, y),
                    (min(start + 7, x0 + 57), y),
                    color,
                    3,
                    cv2.LINE_AA,
                )

        cv2.putText(
            frame,
            label,
            (x0 + 72, y + 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            COLOR_WHITE,
            1,
            cv2.LINE_AA,
        )


def _hud(
    frame, step, goal_distance, prediction_error,
    success, collision, out_of_bounds
):
    """Draw flight telemetry and trajectory legend."""

    overlay = frame.copy()

    cv2.rectangle(
        overlay,
        (25, 22),
        (545, 190),
        COLOR_PANEL,
        -1,
    )

    cv2.addWeighted(
        overlay, 0.82,
        frame, 0.18,
        0, frame
    )

    status = (
        "GOAL REACHED" if success else
        "COLLISION" if collision else
        "OUT OF BOUNDS" if out_of_bounds else
        "ALL CLEAR"
    )

    status_color = (
        (50, 230, 80)
        if status in ("ALL CLEAR", "GOAL REACHED")
        else (40, 40, 240)
    )

    cv2.putText(
        frame,
        "WORLDPILOT | INDOOR AUTONOMOUS FLIGHT",
        (45, 53),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.62,
        COLOR_WHITE,
        2,
        cv2.LINE_AA,
    )

    cv2.putText(
        frame,
        f"STATUS: {status}",
        (45, 91),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.70,
        status_color,
        2,
        cv2.LINE_AA,
    )

    cv2.putText(
        frame,
        f"Step: {step:04d}    Goal distance: {goal_distance:.2f} m",
        (45, 130),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (230, 230, 230),
        1,
        cv2.LINE_AA,
    )

    cv2.putText(
        frame,
        f"World-model prediction error: {prediction_error:.2f} mm",
        (45, 164),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        COLOR_PREDICTED,
        2,
        cv2.LINE_AA,
    )

    _draw_legend(frame)


def render_chase_frame(
    env, actual_path, predicted_path, waypoints, action,
    step, goal_distance, prediction_error,
    success, collision, out_of_bounds, camera_state
):
    """Render PyBullet's real 3D scene from a close, smooth chase camera."""

    position = np.asarray(
        env.state[:3], dtype=np.float64
    )

    # Estimate drone heading from commanded velocity.
    heading = np.asarray(
        action[:2], dtype=np.float64
    )

    norm = np.linalg.norm(heading)

    if norm > 1e-5:
        heading /= norm
        camera_state["heading"] = heading
    else:
        heading = camera_state.get(
            "heading", np.array([1.0, 0.0])
        )

    # Close camera: 0.20 m behind and slightly above drone.
    desired_eye = np.array([
        position[0] - heading[0] * 0.20,
        position[1] - heading[1] * 0.20,
        position[2] + 0.14,
    ])

    desired_target = np.array([
        position[0] + heading[0] * 0.18,
        position[1] + heading[1] * 0.18,
        position[2] + 0.04,
    ])

    client_id = getattr(
        env.base,
        "CLIENT",
        getattr(env.base, "client", 0)
    )

    # Prevent camera from looking through nearby walls.
    ray_start = np.array([
        position[0] - heading[0] * 0.06,
        position[1] - heading[1] * 0.06,
        position[2] + 0.08,
    ])

    hit = p.rayTest(
        ray_start.tolist(),
        desired_eye.tolist(),
        physicsClientId=client_id,
    )[0]

    if hit[0] >= 0 and hit[2] < 0.98:
        fraction = max(0.12, hit[2] - 0.10)

        desired_eye = ray_start + fraction * (
            desired_eye - ray_start
        )

    # Smooth camera movement.
    if camera_state.get("eye") is None:
        camera_state["eye"] = desired_eye.copy()
        camera_state["target"] = desired_target.copy()
    else:
        alpha = 0.13

        camera_state["eye"] += alpha * (
            desired_eye - camera_state["eye"]
        )

        camera_state["target"] += alpha * (
            desired_target - camera_state["target"]
        )

    eye = camera_state["eye"]
    target = camera_state["target"]

    view = p.computeViewMatrix(
        cameraEyePosition=eye.tolist(),
        cameraTargetPosition=target.tolist(),
        cameraUpVector=[0, 0, 1],
    )

    projection = p.computeProjectionMatrixFOV(
        fov=85,
        aspect=WIDTH / HEIGHT,
        nearVal=0.02,
        farVal=40,
    )

    image = p.getCameraImage(
        WIDTH,
        HEIGHT,
        viewMatrix=view,
        projectionMatrix=projection,
        renderer=p.ER_TINY_RENDERER,
        physicsClientId=client_id,
    )

    rgba = np.asarray(
        image[2], dtype=np.uint8
    ).reshape(HEIGHT, WIDTH, 4)

    frame = cv2.cvtColor(
        rgba, cv2.COLOR_RGBA2BGR
    )

    # 1. A* planned route: blue/cyan solid.
    _draw_world_path(
        frame,
        waypoints,
        view,
        projection,
        COLOR_ASTAR,
        thickness=3,
        markers=True,
    )

    # 2. Actual drone trajectory: green solid.
    _draw_world_path(
        frame,
        actual_path,
        view,
        projection,
        COLOR_ACTUAL,
        thickness=3,
    )

    # 3. World-model forecast: orange dashed.
    _draw_dashed_world_path(
        frame,
        predicted_path,
        view,
        projection,
        COLOR_PREDICTED,
        dash_length=9,
        gap_length=7,
        thickness=3,
    )

    _hud(
        frame,
        step,
        goal_distance,
        prediction_error * 1000,
        success,
        collision,
        out_of_bounds,
    )

    return frame, camera_state
