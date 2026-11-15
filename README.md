# WorldPilot
### Learning Predictive Dynamics for Autonomous Drone Navigation

WorldPilot is an experimental robotics project exploring learned dynamics
models for autonomous quadrotors. It combines PyBullet simulation,
classical path planning, and neural state-transition prediction to study
how drones can anticipate their future motion.

The project progresses from open-space obstacle navigation to structured
indoor maze environments.

## Demonstration

**Indoor drone navigation with learned future-state prediction**

The visualization shows three trajectories:

- **Blue:** A* global route.
- **Green:** Actual drone trajectory in PyBullet.
- **Orange dashed:** Neural model's 20-step predicted trajectory.

[Watch the indoor WorldPilot demonstration](outputs/worldpilot_indoor_demo.mp4)

## Architecture

The current system combines three components:

1. **A* planner:** Computes a route through the indoor maze.
2. **Waypoint controller:** Converts waypoints into velocity commands.
3. **Learned dynamics model:** Predicts future drone states from the
   current state and commanded action.

PyBullet executes the actual flight, allowing predicted and observed
trajectories to be compared.

```text
        Indoor Maze
             |
             v
        A* Planner
             |
             v
    Waypoint Controller
             |
             v
      PyBullet Drone
             |
       Actual State
             |
             v
   Learned Dynamics Model
             |
             v
    Predicted Trajectory
```

## Learned Dynamics Model

WorldPilot currently uses a residual multilayer perceptron (MLP) to
predict drone state transitions:

\[
\hat{s}_{t+1}=s_t+f_\theta(s_t,a_t)
\]

Where:
- State: 28 dimensions (position, velocity, attitude, goal and obstacles).
- Action: 3-dimensional world-frame velocity command.
- Output: Predicted change in the drone state.

Predictions are fed back into the model for autoregressive rollouts.

## Experimental Results

### Open-space environment

| Metric | Result |
|---|---:|
| Episodes | 40 |
| Transitions | 11,465 |
| Goal-reaching episodes | 40 |
| One-step position RMSE | 0.338 mm |
| 5-step position RMSE | 2.91 mm |
| 10-step position RMSE | 7.31 mm |
| 20-step position RMSE | 23.21 mm |
| 30-step position RMSE | 40.75 mm |

### Indoor maze environment

| Metric | Result |
|---|---:|
| Episodes | 20 |
| Transitions | 28,847 |
| State dimension | 28 |
| Action dimension | 3 |
| One-step position RMSE | 0.304 mm |
| 5-step position RMSE | 0.881 mm |
| 10-step position RMSE | 1.458 mm |
| 20-step position RMSE | 2.786 mm |

Results are from episode-level held-out evaluations. The indoor experiments
use a fixed maze and a consistent route with noisy control commands.
They do not establish generalization to unseen maze layouts.

## Running the Project

Create an environment and install dependencies:

```bash
conda create -n worldpilot python=3.12 -y
conda activate worldpilot
pip install -r requirements.txt
```

Run the simulation test:

```bash
python scripts/smoke_test.py
```

Train the indoor dynamics model:

```bash
python scripts/train_world_model.py \
  --data datasets/worldpilot_maze.npz \
  --output checkpoints/worldpilot_maze_mlp.pt \
  --epochs 100 --seed 42
```

Generate the indoor predictive demonstration:

```bash
python scripts/run_maze_world_model_demo.py \
  --output outputs/worldpilot_indoor_demo.mp4
```

Dataset and checkpoint files are generated locally and are not included
in the repository by default.

## Current Limitations

- A* and waypoint following handle navigation; the learned model does
  not yet select actions.
- The indoor evaluation uses a fixed maze and route.
- The model has not been validated on a physical drone.
- Visual perception and uncertainty-aware prediction are not yet implemented.

## Research Direction

The next steps are to develop more capable predictive models for
autonomous navigation:

- Environment-aware spatial representations.
- Generalization across unseen environments.
- Model-predictive control using learned dynamics.
- Uncertainty-aware trajectory prediction.
- Safety-aware planning with Control Barrier Functions (CBFs).

The long-term goal is to move from predicting drone motion toward
using learned world models for safer, more capable autonomous systems.

## Project Structure

```text
WorldPilot/
├── configs/       # Environment configurations
├── scripts/       # Data collection, training and demonstrations
├── worldpilot/    # Simulation and environment modules
├── datasets/      # Locally generated datasets
├── checkpoints/   # Trained model weights
├── outputs/       # Videos and evaluation visualizations
├── requirements.txt
└── README.md
```

## Acknowledgements

Built using PyBullet, gym-pybullet-drones, PyTorch and NumPy.
