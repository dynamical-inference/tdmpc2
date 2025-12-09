import sys
from termios import TAB0
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
override_cfg = dict(
    obs='state',
    seed=1,
    compile=False,
    mpc=True,
    save_video=True,
    record_planning=False,
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
BASE_SAVE_DIR = f'logs/25-12-09-cartpole_initial_conditions_seeds_models'

# SEEDS FOR THE SWEEP
SEEDS = [1, 2, 3, 4, 5]

MULTITASK_CONFIGS = []
MULTITASK_MODEL_SIZES = [48, 317]
for seed in SEEDS:
    for model_size in MULTITASK_MODEL_SIZES:
        for initial_state in INITIAL_STATES:
            MULTITASK_CONFIGS.append({
                'initial_state': initial_state,
                'model_size': model_size,
                'multitask': True,
                'model_size': model_size,
                'seed': seed,
            })

SINGLE_TASK_CONFIGS = []
for seed in SEEDS:
    for initial_state in INITIAL_STATES:
        SINGLE_TASK_CONFIGS.append({
            'initial_state': initial_state,
            'multitask': False,
            'seed': seed,
        })

CONFIGS = MULTITASK_CONFIGS + SINGLE_TASK_CONFIGS

# ============================================================================
# MAIN SWEEP
# ============================================================================


def main():
    print("=" * 80)
    print("INITIAL CONDITIONS SEEDS SWEEP")
    print("=" * 80)
    print(f"Total initial states: {len(INITIAL_STATES)}")
    print(f"Seeds: {SEEDS}")
    print(f"Base save directory: {BASE_SAVE_DIR}")
    print("=" * 80)

    episode_count = 0
    for config in CONFIGS:
        episode_count += 1

        # Initialize Hydra config
        with initialize(config_path="../tdmpc2", version_base=None):
            cfg = compose(config_name="config")
            OmegaConf.set_struct(cfg, False)
            cfg = OmegaConf.merge(cfg, override_cfg)

        if config['multitask']:
            cfg.task = 'mt30'
            cfg.model_size = config['model_size']
            cfg.multitask = True
            cfg.checkpoint = f'ckpts/mt30-{config["model_size"]}M.pt'
            TASK_IDX = 10  # cartpole-swingup is at index 10 in the mt30 task list
        else:
            cfg.task = 'cartpole-swingup'
            cfg.multitask = False
            cfg.checkpoint = f'ckpts/cartpole-swingup-3.pt'
            TASK_IDX = None

        config['task_name'] = 'cartpole-swingup'

        # Parse config
        cfg = parse_cfg(cfg)

        seed = config['seed']
        initial_state = config['initial_state']
        cfg.initial_state = {
            'qpos': {
                'slider': initial_state[0],
                'hinge_1': initial_state[1],
            },
        }
        # Set seed for this episode
        set_seed(seed)

        # Initialize environment and agent
        env, agent = setup_agent(cfg)

        # Create episode-specific save directory
        time = datetime.now().strftime("%Y%m%d_%H%M%S")
        episode_dir = Path(BASE_SAVE_DIR) / time
        activations_dir = episode_dir / 'activations'
        activations_dir.mkdir(parents=True, exist_ok=True)

        # Create episode recorder to save full episode data
        episode_recorder = EpisodeDataRecorder(cfg,
                                               save_dir=str(activations_dir))

        # Run episode without any interventions
        run_episode_with_recording(env=env,
                                   agent=agent,
                                   episode_recorder=episode_recorder,
                                   planning_recorder=None,
                                   save_video=cfg.save_video,
                                   eval_mode=True,
                                   task=TASK_IDX,
                                   task_name=config['task_name'],
                                   episode_dir=str(episode_dir))

        # Save episode data with metadata
        metadata = {
            'seed': seed,
            'initial_state': initial_state.tolist(),
            'checkpoint': str(cfg.checkpoint),
            'task': cfg.task,
            'task_name': config['task_name'],
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
