import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, "/home/hgf_hmgu/hgf_gib4562/tdmpc2/tdmpc2")

from hydra import initialize, compose
from omegaconf import OmegaConf
from utils import setup_agent, run_episode_with_recording
from common.episode_data_recorder import EpisodeDataRecorder
from common.seed import set_seed
from common.parser import parse_cfg
from datetime import datetime
import pickle
import torch
# ============================================================================
# SWEEP CONFIGURATION
# ============================================================================

# Base configuration
model_size = 317
CKPT_PATH = f'ckpts/mt30-{model_size}M.pt'
override_cfg = dict(
    task='mt30',  # Must match checkpoint (was 'cartpole-swingup')
    checkpoint=CKPT_PATH,
    obs='state',
    seed=1,
    compile=False,
    mpc=True,
    multitask=True,
    model_size=model_size,  # Must match checkpoint (was 5)
    save_video=False,  #NOTE (R): for debugging purposes
    record_planning=False,  # Focus on episode data
)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

negative_range = np.linspace(-3.14, -1.5, 5)  # 5 episodes in the negative range
positive_range = np.linspace(1.5, 3.14, 5)  # 4 episodes in the positive range
HINGE_1_VALUES = np.concatenate([negative_range, positive_range]).round(2)
BASE_SAVE_DIR = f'logs/181125-initial_conditions_mt30-{model_size}M_sweep'

# SEEDS FOR THE SWEEP
SEEDS = [1, 2, 3, 4, 5]

CONFIGS = []
for seed in SEEDS:
    for hinge_1 in HINGE_1_VALUES:
        CONFIGS.append({
            'hinge_1_value': hinge_1,
            'seed': seed,
        })

# ============================================================================
# MAIN SWEEP
# ============================================================================


def main():
    print("=" * 80)
    print("INITIAL CONDITIONS MT30-48M SWEEP")
    print("=" * 80)
    print(f"Task: {override_cfg['task']}")
    print(f"Checkpoint: {override_cfg['checkpoint']}")
    print(f"Total configurations: {len(CONFIGS)}")
    print(f"Hinge 1 values: {HINGE_1_VALUES}")
    print(f"Base save directory: {BASE_SAVE_DIR}")
    print("=" * 80)

    # Initialize Hydra config
    with initialize(config_path="../tdmpc2", version_base=None):
        cfg = compose(config_name="config")
        OmegaConf.set_struct(cfg, False)
        cfg = OmegaConf.merge(cfg, override_cfg)

    # Parse config
    cfg = parse_cfg(cfg)
    set_seed(cfg.seed)

    # Run sweep
    print("\n" + "=" * 80)
    print("RUNNING SWEEP")
    print("=" * 80 + "\n")
    TASK_IDX = 10  # cartpole-swingup is at index 10 in the mt30 task list

    for i, config in enumerate(CONFIGS):

        if config['hinge_1_value'] == -3.14:
            continue

        # Initialize environment and agent
        print("\nInitializing environment and agent...")
        hinge_1 = 1.5  #config['hinge_1_value']
        cfg.initial_state = {
            'qpos': {
                'hinge_1': float(hinge_1),
            },
        }
        env, agent = setup_agent(cfg)

        # Create episode-specific save directory
        time = datetime.now().strftime("%Y%m%d_%H%M%S")
        episode_dir = Path(BASE_SAVE_DIR) / time
        activations_dir = episode_dir / 'activations'
        activations_dir.mkdir(parents=True, exist_ok=True)

        print(f"Running episode for task: {cfg.task} with index: {TASK_IDX}")
        # Create episode recorder to save full episode data
        episode_recorder = EpisodeDataRecorder(cfg,
                                               save_dir=str(activations_dir))

        # Run episode with recording (reuses shared function from analyze_activations.py)
        # cartpole-swingup is at index 10 in the mt30 task list
        run_episode_with_recording(env=env,
                                   agent=agent,
                                   episode_recorder=episode_recorder,
                                   planning_recorder=None,
                                   save_video=cfg.save_video,
                                   eval_mode=True,
                                   task=TASK_IDX,
                                   episode_dir=episode_dir)

        metadata = {
            'config': config,
            'checkpoint': str(cfg.checkpoint),
            'task': cfg.task,
            'latent_dim': cfg.latent_dim,
        }
        episode_recorder.save_episode(metadata=metadata)
        print(f"  → Saved episode data to: {activations_dir}")


print("\n" + "=" * 80)
print("SWEEP COMPLETE!")
print("=" * 80)
print(f"Results saved to: {BASE_SAVE_DIR}")

if __name__ == '__main__':
    main()
