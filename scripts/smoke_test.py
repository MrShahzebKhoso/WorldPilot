"""Fast headless smoke test: environment reset, step, and state/action dimensions."""
import numpy as np
from worldpilot.config import load_config
from worldpilot.environment import WorldPilotDroneEnv


def main():
    config = load_config()
    env = WorldPilotDroneEnv(config, gui=False, seed=123)
    try:
        state, _ = env.reset(seed=123)
        assert state.shape == (28,), state.shape
        for _ in range(20):
            action = env.waypoint_controller(state, exploration_sigma=0.0)
            next_state, _, done, info = env.step(action)
            assert next_state.shape == (28,)
            assert np.isfinite(next_state).all()
            assert action.shape == (3,)
            state = next_state
            if done:
                break
        print("SMOKE TEST PASSED")
        print(f"State dimension: {env.STATE_DIM}")
        print(f"Action dimension: {env.ACTION_DIM}")
        print(f"Last position: {state[:3].round(3).tolist()}")
    finally:
        env.close()


if __name__ == "__main__":
    main()
