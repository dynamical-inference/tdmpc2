import sys
import numpy as np
from pathlib import Path
from itertools import product

sys.path.insert(0, "/home/hgf_hmgu/hgf_gib4562/tdmpc2/tdmpc2")

from hydra import initialize, compose
from omegaconf import OmegaConf
from utils import setup_agent, run_episode_with_recording
from common.episode_data_recorder import EpisodeDataRecorder
from common.seed import set_seed
from common.parser import parse_cfg
from datetime import datetime
import torch

# ============================================================================
# SWEEP CONFIGURATION
# ============================================================================

# Models directory
MODELS_DIR = Path('logs/model-runs/2025-12-20/19-10-12/cartpole_exp/models')

# Select which model checkpoints to use (by training step)
# Available: 25000, 50000, 75000, ..., 3550000 (every 25000 steps)
# Example: use a subset of checkpoints at different training stages
MODEL_STEPS = [
    25_000,  # Early training
    50_000,  # Mid-early training
    100_000,  # Mid training
    500_000,  # Late training
    1_000_000,  # Near convergence
    3_000_000,  # Final checkpoint
]

# Base configuration (checkpoint will be overwritten per model)
override_cfg = dict(
    task='cartpole-swingup',  # Must match checkpoint task
    checkpoint=None,  # Will be set per model
    obs='state',
    seed=1,
    compile=False,
    mpc=True,
    multitask=False,
    model_size=
    5,  # Must match checkpoint (from config.yaml this was left as ???, adjust if needed)
    save_video=True,  # NOTE (R): for debugging purposes
    record_planning=False,  # Focus on episode data
)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# SEEDS FOR THE SWEEP
SEEDS = [1]

# ============================================================================
# GENERATE INITIAL STATES
# ============================================================================

# Define the ranges for each dimension
cart_positions = [-1, 0, 1]
angles_degrees = np.linspace(0, 360, 6)
angles_radians = np.deg2rad(angles_degrees)

# Generate all combinations
all_combinations = []
for cart_pos, angle_radians in product(
        cart_positions,
        angles_radians,
):
    all_combinations.append([cart_pos, angle_radians])

# Convert to numpy array
INITIAL_STATES = np.array(all_combinations)

# Base directory for saving results
BASE_SAVE_DIR = 'logs/26-01-04-cartpole_swingup-model_sweep'

# ============================================================================
# MAIN SWEEP
# ============================================================================


def get_model_path(step: int) -> Path:
    """Get the path to a model checkpoint by training step."""
    model_path = MODELS_DIR / f"{step}.pt"
    if not model_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {model_path}")
    return model_path


def main():
    print("=" * 80)
    print("MODEL SWEEP - INITIAL CONDITIONS AND SEEDS")
    print("=" * 80)
    print(f"Task: {override_cfg['task']}")
    print(f"Models directory: {MODELS_DIR}")
    print(f"Model checkpoints: {MODEL_STEPS}")
    print(f"Total initial states: {len(INITIAL_STATES)}")
    print(f"Seeds: {SEEDS}")
    print(f"Total episodes per model: {len(INITIAL_STATES) * len(SEEDS)}")
    print(
        f"Total episodes overall: {len(MODEL_STEPS) * len(INITIAL_STATES) * len(SEEDS)}"
    )
    print(f"Base save directory: {BASE_SAVE_DIR}")
    print("=" * 80)

    # Verify all model checkpoints exist
    print("\nVerifying model checkpoints...")
    for step in MODEL_STEPS:
        model_path = get_model_path(step)
        print(f"  ✓ Found: {model_path.name}")

    # Initialize Hydra config
    with initialize(config_path="../tdmpc2", version_base=None):
        cfg = compose(config_name="config")
        OmegaConf.set_struct(cfg, False)
        cfg = OmegaConf.merge(cfg, override_cfg)

    # Parse config
    cfg = parse_cfg(cfg)

    # Run sweep
    print("\n" + "=" * 80)
    print("RUNNING SWEEP")
    print("=" * 80 + "\n")

    total_episodes = len(MODEL_STEPS) * len(INITIAL_STATES) * len(SEEDS)
    episode_count = 0

    for model_step in MODEL_STEPS:
        model_path = get_model_path(model_step)
        print("\n" + "=" * 80)
        print(f"MODEL: {model_path.name} (step {model_step})")
        print("=" * 80)

        # Update checkpoint path in config
        cfg.checkpoint = str(model_path)

        for seed in SEEDS:
            for initial_state in INITIAL_STATES:
                episode_count += 1
                print(f"\n[Episode {episode_count}/{total_episodes}]")
                print(f"  Model: {model_path.name}")
                print(f"  Seed: {seed}, Initial state: {initial_state}")

                # Set seed for this episode
                set_seed(seed)

                cfg.initial_state = {
                    'qpos': {
                        'slider': initial_state[0],  # cart position
                        'hinge_1': initial_state[1],  # pole angle
                    },
                    'qvel': {
                        'slider': 0.,  # cart velocity
                        'hinge_1': 0.,  # pole angular velocity
                    },
                }

                # Initialize environment and agent
                env, agent = setup_agent(cfg)

                # Create episode-specific save directory
                # Organize by model step for easier analysis
                time = datetime.now().strftime("%Y%m%d_%H%M%S")
                episode_dir = Path(BASE_SAVE_DIR) / time
                activations_dir = episode_dir / 'activations'
                activations_dir.mkdir(parents=True, exist_ok=True)

                # Create episode recorder to save full episode data
                episode_recorder = EpisodeDataRecorder(
                    cfg, save_dir=str(activations_dir))

                # Run episode without any interventions
                run_episode_with_recording(env=env,
                                           agent=agent,
                                           episode_recorder=episode_recorder,
                                           planning_recorder=None,
                                           save_video=cfg.save_video,
                                           eval_mode=True,
                                           task=None,
                                           task_name=cfg.task,
                                           episode_dir=str(episode_dir))

                # Save episode data with metadata
                metadata = {
                    'seed': seed,
                    'initial_state': initial_state.tolist(),
                    'checkpoint': str(cfg.checkpoint),
                    'model_step': model_step,
                    'task': cfg.task,
                }
                episode_recorder.save_episode(metadata=metadata)
                print(f"  → Saved episode data to: {activations_dir}")

    print("\n" + "=" * 80)
    print("SWEEP COMPLETE!")
    print("=" * 80)
    print(f"Total episodes run: {episode_count}")
    print(f"Results saved to: {BASE_SAVE_DIR}")


if __name__ == '__main__':
    main()
