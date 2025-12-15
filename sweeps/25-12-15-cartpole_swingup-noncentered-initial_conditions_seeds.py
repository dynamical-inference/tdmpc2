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

# Base configuration
CKPT_PATH = f'ckpts/cartpole-swingup-noncentered.pt'
override_cfg = dict(
    task=
    'cartpole-swingup_custom',  # Must match checkpoint (was 'cartpole-swingup')
    checkpoint=CKPT_PATH,
    obs='state',
    seed=1,
    compile=False,
    mpc=True,
    multitask=False,
    model_size=5,  # Must match checkpoint (was 5)
    save_video=True,  #NOTE (R): for debugging purposes
    record_planning=False,  # Focus on episode data
)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# SEEDS FOR THE SWEEP
SEEDS = [1, 2, 3, 4, 5]

# ============================================================================
# GENERATE INITIAL STATES
# ============================================================================

# Define the ranges for each dimension
cart_positions = [-1.5, -1, 0, 1, 1.5]
angles_degrees = np.linspace(
    0, 360, 11)  # Generate 20 equidistant angles from 0 to 360 degrees
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
#date_logged = datetime.now().strftime("%d%m%Y")
BASE_SAVE_DIR = f'logs/25-12-15-cartpole_swingup-noncentered-initial_conditions_seeds_sweep'

# ============================================================================
# MAIN SWEEP
# ============================================================================


def main():
    print("=" * 80)
    print("INITIAL CONDITIONS SEEDS SWEEP")
    print("=" * 80)
    print(f"Task: {override_cfg['task']}")
    print(f"Checkpoint: {override_cfg['checkpoint']}")
    print(f"Total initial states: {len(INITIAL_STATES)}")
    print(f"Seeds: {SEEDS}")
    print(f"Total episodes: {len(INITIAL_STATES) * len(SEEDS)}")
    print(f"Base save directory: {BASE_SAVE_DIR}")
    print("=" * 80)

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

    episode_count = 0
    for seed in SEEDS:
        for initial_state in INITIAL_STATES:
            episode_count += 1
            print(
                f"\n[Episode {episode_count}/{len(INITIAL_STATES) * len(SEEDS)}]"
            )
            print(f"  Seed: {seed}, Initial state: {initial_state}")

            # Set seed for this episode
            set_seed(seed)

            cfg.initial_state = {
                'qpos': {
                    'slider': initial_state[0],  #cart position
                    'hinge_1': initial_state[1],  # pole angle
                },
                # 'qvel': {
                #     'slider': initial_state[2],  # cart velocity
                #     'hinge_1': initial_state[3],  # pole angular velocity
                # },
            }

            # Initialize environment and agent
            env, agent = setup_agent(cfg)

            # Create episode-specific save directory
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
