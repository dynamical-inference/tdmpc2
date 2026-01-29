"""
Sweep for evaluating ProjectedTDMPC2 with different projection methods and dimensions
across various initial conditions.

This sweep:
1. Loads pre-fitted projectors from disk (PCA, hybrid OLS, CCA, nonlinear)
2. Evaluates the projected agent across initial conditions
3. Records episode data (observations, actions, rewards, latents) for analysis
"""

import sys
import numpy as np
from pathlib import Path
from itertools import product

sys.path.insert(0, "/home/hgf_hmgu/hgf_gib4562/tdmpc2/tdmpc2")

from hydra import initialize, compose
from omegaconf import OmegaConf
from utils import run_episode_with_recording
from common.episode_data_recorder import EpisodeDataRecorder
from common.latent_projector import LatentProjector
from common.nonlinear_latent_projector import NonlinearLatentProjector
from common.seed import set_seed
from common.parser import parse_cfg
from envs import make_env
from tdmpc2 import TDMPC2
from projected_tdmpc2 import ProjectedTDMPC2
from datetime import datetime
import torch

# ============================================================================
# SWEEP CONFIGURATION
# ============================================================================

# Model checkpoint to use
CHECKPOINT_PATH = Path(
    'logs/model-runs/2025-12-20/19-10-12/cartpole_exp/models/500000.pt')

# Directory containing pre-fitted projectors
PROJECTORS_DIR = Path(
    'sweeps/sweep_data/26-01-23-cartpole_swingup-projected_sweep_uniform_data')

# Also run baseline (no projection) for comparison
RUN_BASELINE = False

# Base configuration
override_cfg = dict(
    task='cartpole-swingup',
    checkpoint=str(CHECKPOINT_PATH),
    obs='state',
    seed=1,
    compile=False,
    mpc=True,
    multitask=False,
    model_size=5,
    save_video=True,  # Save video for each episode
    record_planning=False,
)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Seeds for the sweep
SEEDS = [1]

# ============================================================================
# PROJECTOR CONFIGURATIONS
# ============================================================================


def get_projector_configs():
    """
    Define all projector configurations to sweep over.
    
    Returns list of dicts with:
        - name: Human-readable name for the config
        - path: Path to the projector file
        - type: 'linear' or 'nonlinear'
    """
    configs = []

    # PCA projectors: k = 8, 9, 10, 11, 12, 13, 14, 15
    for k in [8, 9, 10, 11, 12, 13, 14, 15]:
        configs.append({
            'name': f'pca_k{k}',
            'path': PROJECTORS_DIR / f'pca_projector_k{k}.pt',
            'type': 'linear',
            'method': 'pca',
            'k': k,
        })

    # Hybrid OLS projectors: k_unsupervised = 4-10, k_supervised = 5
    k_supervised = 5
    for k_unsupervised in [9, 10, 15]:  # 5, 6, 7, 8,
        k_total = k_unsupervised + k_supervised
        configs.append({
            'name':
                f'hybrid_ols_k{k_total}_ksup{k_supervised}',
            'path':
                PROJECTORS_DIR /
                f'hybrid_ols_projector_k{k_unsupervised}_k_supervised{k_supervised}.pt',
            'type':
                'linear',
            'method':
                'hybrid_ols',
            'k':
                k_total,
            'k_supervised':
                k_supervised,
            'k_unsupervised':
                k_unsupervised,
        })

    # CCA projector: k = 5
    configs.append({
        'name': 'cca_k5',
        'path': PROJECTORS_DIR / 'cca_projector_k5.pt',
        'type': 'linear',
        'method': 'cca',
        'k': 5,
    })

    # Nonlinear projectors: k = 3, 4, 5, 6, 7, 8, 10, 15
    for k in [1, 2, 3, 4, 5, 6, 7, 8, 10, 15]:
        configs.append({
            'name': f'nonlinear_k{k}',
            'path': PROJECTORS_DIR / f'nonlinear_projector_k{k}.pt',
            'type': 'nonlinear',
            'method': 'nonlinear',
            'k': k,
        })

    for k in [2, 3, 4, 10]:
        configs.append({
            'name':
                f'state_supervised_nonlinear_k{k}',
            'path':
                PROJECTORS_DIR /
                f'state_supervised_nonlinear_projector_k{k}.pt',
            'type':
                'nonlinear',
            'method':
                'nonlinear',
            'k':
                k,
        })
    return configs


def load_projector(config):
    """Load a projector from disk based on config."""
    path = config['path']
    if not path.exists():
        raise FileNotFoundError(
            f"Projector not found: {path}\n"
            f"Please fit projectors first and save them to {PROJECTORS_DIR}")

    if config['type'] == 'linear':
        return LatentProjector.load(str(path))
    else:  # nonlinear
        return NonlinearLatentProjector.load(str(path))


# ============================================================================
# GENERATE INITIAL STATES
# ============================================================================

# Define the ranges for each dimension
cart_positions = [0]
angles_degrees = np.linspace(0, 360, 9)[:-1]
angles_radians = np.deg2rad(angles_degrees)

# Generate all combinations
all_combinations = []
for cart_pos, angle_radians in product(cart_positions, angles_radians):
    all_combinations.append([cart_pos, angle_radians])

# Convert to numpy array
INITIAL_STATES = np.array(all_combinations)

# Base directory for saving results
BASE_SAVE_DIR = 'logs/26-01-23-cartpole_swingup-projected_sweep_uniform_data'

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================


def setup_projected_agent(cfg, projector):
    """
    Initialize environment and projected agent.
    
    Args:
        cfg: Configuration object
        projector: LatentProjector or NonlinearLatentProjector instance
        
    Returns:
        env: Environment instance
        agent: ProjectedTDMPC2 agent instance
    """
    env = make_env(cfg)
    agent = ProjectedTDMPC2(cfg, projector=projector)

    if cfg.checkpoint and cfg.checkpoint != '???':
        agent.load(cfg.checkpoint)
    else:
        raise ValueError("Must provide a checkpoint path")

    return env, agent


def setup_baseline_agent(cfg):
    """
    Initialize environment and baseline (non-projected) agent.
    Uses RecordingTDMPC2 for compatibility with run_episode_with_recording.
    """
    from recording_tdmpc2 import RecordingTDMPC2

    env = make_env(cfg)
    agent = RecordingTDMPC2(cfg, planning_recorder=None)

    if cfg.checkpoint and cfg.checkpoint != '???':
        agent.load(cfg.checkpoint)
    else:
        raise ValueError("Must provide a checkpoint path")

    return env, agent


# ============================================================================
# MAIN SWEEP
# ============================================================================


def main():
    # Get all projector configurations
    projector_configs = get_projector_configs()

    print("=" * 80)
    print("PROJECTED TDMPC2 SWEEP")
    print("=" * 80)
    print(f"Task: {override_cfg['task']}")
    print(f"Checkpoint: {CHECKPOINT_PATH}")
    print(f"Projectors directory: {PROJECTORS_DIR}")
    print(f"Run baseline: {RUN_BASELINE}")
    print(f"Total initial states: {len(INITIAL_STATES)}")
    print(f"Seeds: {SEEDS}")
    print(f"Base save directory: {BASE_SAVE_DIR}")
    print(f"\nProjector configurations ({len(projector_configs)} total):")
    for cfg_proj in projector_configs:
        print(f"  - {cfg_proj['name']} ({cfg_proj['type']}, k={cfg_proj['k']})")
    print("=" * 80)

    # Verify checkpoint exists
    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(f"Checkpoint not found: {CHECKPOINT_PATH}")
    print(f"✓ Found checkpoint: {CHECKPOINT_PATH}")

    # Verify and load projectors
    print("\nVerifying and loading projectors...")
    projectors = {}
    for cfg_proj in projector_configs:
        try:
            projectors[cfg_proj['name']] = load_projector(cfg_proj)
            print(f"  ✓ Loaded: {cfg_proj['name']} ({cfg_proj['path'].name})")
        except FileNotFoundError as e:
            print(f"  ✗ Missing: {cfg_proj['name']} ({cfg_proj['path'].name})")
            raise e

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

    # Determine what to sweep over (projectors first, then baseline)
    sweep_configs = []

    for cfg_proj in projector_configs:
        sweep_configs.append({
            'name': cfg_proj['name'],
            'projector': projectors[cfg_proj['name']],
            'metadata': {
                'name': cfg_proj['name'],
                'path': str(cfg_proj['path']),  # Convert Path to string here
                'type': cfg_proj['type'],
                'method': cfg_proj['method'],
                'k': cfg_proj['k'],
                'k_supervised': cfg_proj.get('k_supervised'),
                'k_unsupervised': cfg_proj.get('k_unsupervised'),
            },
        })

    # Add baseline at the end
    if RUN_BASELINE:
        sweep_configs.append({
            'name': 'baseline',
            'projector': None,
            'metadata': {
                'name': 'baseline',
                'type': 'none',
                'method': 'none',
                'k': None,
            }
        })

    total_episodes = len(sweep_configs) * len(INITIAL_STATES) * len(SEEDS)
    episode_count = 0

    for sweep_cfg in sweep_configs:
        config_name = sweep_cfg['name']
        projector = sweep_cfg.get('projector')
        proj_metadata = sweep_cfg.get('metadata', {})

        print("\n" + "=" * 80)
        print(f"CONFIG: {config_name}")
        print("=" * 80)

        for seed in SEEDS:
            for initial_state in INITIAL_STATES:
                episode_count += 1
                print(f"\n[Episode {episode_count}/{total_episodes}]")
                print(f"  Config: {config_name}")
                print(f"  Seed: {seed}, Initial state: {initial_state}")

                # Set seed for this episode
                set_seed(seed)

                # Set initial state
                cfg.initial_state = {
                    'qpos': {
                        'slider': initial_state[0],
                        'hinge_1': initial_state[1],
                    },
                    'qvel': {
                        'slider': 0.,
                        'hinge_1': 0.,
                    },
                }

                # Create environment and agent
                if projector is None:
                    # Baseline: use RecordingTDMPC2
                    env, agent = setup_baseline_agent(cfg)
                else:
                    # Projected agent
                    env, agent = setup_projected_agent(cfg, projector)

                # Create episode-specific save directory
                time = datetime.now().strftime("%Y%m%d_%H%M%S")
                episode_dir = Path(BASE_SAVE_DIR) / time
                activations_dir = episode_dir / 'activations'
                activations_dir.mkdir(parents=True, exist_ok=True)

                # Create episode recorder
                episode_recorder = EpisodeDataRecorder(
                    cfg, save_dir=str(activations_dir))

                # Run episode using the existing utility function
                run_episode_with_recording(
                    env=env,
                    agent=agent,
                    episode_recorder=episode_recorder,
                    planning_recorder=None,
                    save_video=cfg.save_video,
                    eval_mode=True,
                    task=None,
                    task_name=cfg.task,
                    episode_dir=str(episode_dir),
                )

                # Save episode data with metadata
                metadata = {
                    'seed': seed,
                    'initial_state': initial_state.tolist(),
                    'checkpoint': str(cfg.checkpoint),
                    'task': cfg.task,
                    'config_name': config_name,
                    'projector_config': proj_metadata,  # Already serializable
                }
                episode_recorder.save_episode(metadata=metadata)
                print(f"  → Saved to: {activations_dir}")

    print("\n" + "=" * 80)
    print("SWEEP COMPLETE!")
    print("=" * 80)
    print(f"Total episodes run: {episode_count}")
    print(f"Results saved to: {BASE_SAVE_DIR}")


if __name__ == '__main__':
    main()
