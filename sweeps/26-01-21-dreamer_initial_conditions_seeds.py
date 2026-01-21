"""
Sweep script for DreamerV3 agent initial conditions experiment.

This script runs the DreamerV3 agent across different initial conditions
(cart positions and pole angles) and records activations, observations,
and reconstructions for mechanistic interpretability analysis.

Similar to the TD-MPC2 sweep in 25-11-28-cartpole_initial_conditions_seeds.py
but adapted for DreamerV3's architecture (RSSM, encoder/decoder, etc.).
"""

import sys
import os
import numpy as np
from pathlib import Path
from itertools import product
from datetime import datetime
import argparse

# Add paths
DREAMER_PATH = Path(__file__).parent.parent / "dreamerv3-torch"
sys.path.insert(0, str(DREAMER_PATH))

import torch
import ruamel.yaml as yaml

from episode_data_recorder import DreamerEpisodeDataRecorder
from recording_dreamer import setup_dreamer_agent, run_dreamer_episode_with_recording
import tools

# ============================================================================
# LOAD CONFIG FROM YAML
# ============================================================================


def load_config_from_yaml(config_names=None):
    """
    Load configuration from dreamerv3-torch/configs.yaml.
    
    Args:
        config_names: List of config names to merge (e.g., ['defaults', 'dmc_vision'])
                     If None, uses ['defaults']
    
    Returns:
        dict: Merged configuration dictionary
    """
    config_path = DREAMER_PATH / "configs.yaml"
    configs = yaml.safe_load(config_path.read_text())

    def recursive_update(base, update):
        for key, value in update.items():
            if isinstance(value, dict) and key in base:
                recursive_update(base[key], value)
            else:
                base[key] = value

    if config_names is None:
        config_names = ["defaults"]

    merged_config = {}
    for name in config_names:
        recursive_update(merged_config, configs[name])

    return merged_config


# Load default config from YAML
# Use dmc_proprio config (state-based, not vision)
DEFAULT_CONFIG = load_config_from_yaml(["defaults", "dmc_proprio"])
# Override task to cartpole_swingup
DEFAULT_CONFIG['task'] = 'dmc_cartpole_swingup'
# Use deterministic evaluation (mean of stochastic state)
DEFAULT_CONFIG['eval_state_mean'] = True
# Larger video resolution
DEFAULT_CONFIG['size'] = [256, 256]

# Debug: verify encoder config
print(f"Loaded encoder config: {DEFAULT_CONFIG['encoder']}")
print(f"Loaded decoder config: {DEFAULT_CONFIG['decoder']}")

# Device
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Seeds for the sweep
SEEDS = [1, 2, 3, 4, 5]

# ============================================================================
# GENERATE INITIAL STATES
# ============================================================================

# Define the ranges for each dimension (cartpole swingup)
cart_positions = [0]
angles_degrees = [
    136, 180, 224
]  #np.linspace(0, 360,11)  # 11 equidistant angles from 0 to 360 degrees
angles_radians = np.deg2rad(angles_degrees)

# Generate all combinations
all_combinations = []
for cart_pos, angle_radians in product(cart_positions, angles_radians):
    all_combinations.append([cart_pos, angle_radians])

# Convert to numpy array
INITIAL_STATES = np.array(all_combinations)


class ConfigNamespace:
    """Simple namespace class to mimic argparse namespace.
    
    Note: Nested dicts are kept as dicts (not converted to ConfigNamespace)
    because DreamerV3's model code uses **config.encoder, **config.decoder, etc.
    which requires actual dict objects.
    """

    def __init__(self, config_dict):
        for key, value in config_dict.items():
            # Keep dicts as dicts (don't convert to ConfigNamespace)
            # This is needed because DreamerV3 uses **config.encoder, etc.
            setattr(self, key, value)

    def get(self, key, default=None):
        return getattr(self, key, default)

    def __contains__(self, key):
        return hasattr(self, key)


def create_config(checkpoint_path, seed, initial_state=None):
    """Create configuration for a single run."""
    config = DEFAULT_CONFIG.copy()
    config['device'] = str(DEVICE)
    config['seed'] = seed

    # Set initial state for cartpole
    if initial_state is not None:
        config['initial_state'] = {
            'qpos': {
                'slider': initial_state[0],  # cart position
                'hinge_1': initial_state[1],  # pole angle
            },
        }
    else:
        config['initial_state'] = None

    # config
    # Convert to namespace
    cfg = ConfigNamespace(config)

    return cfg


def main():
    parser = argparse.ArgumentParser(
        description='DreamerV3 Initial Conditions Sweep')
    parser.add_argument('--checkpoint',
                        type=str,
                        required=True,
                        help='Path to DreamerV3 checkpoint')
    parser.add_argument('--save_dir',
                        type=str,
                        default=None,
                        help='Base directory for saving results')
    parser.add_argument('--no_video',
                        action='store_true',
                        default=False,
                        help='Disable saving episode videos')
    parser.add_argument('--record_reconstructions',
                        action='store_true',
                        default=True,
                        help='Record decoder reconstructions')
    parser.add_argument('--max_episodes',
                        type=int,
                        default=None,
                        help='Maximum number of episodes to run (for testing)')
    args = parser.parse_args()

    # Set up save directory
    if args.save_dir is None:
        date_str = datetime.now().strftime("%Y%m%d")
        args.save_dir = f'logs/{date_str}-dreamer_initial_conditions_sweep'

    BASE_SAVE_DIR = Path(args.save_dir)
    BASE_SAVE_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("DREAMERV3 INITIAL CONDITIONS SWEEP")
    print("=" * 80)
    print(f"Task: dmc_cartpole_swingup")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Total initial states: {len(INITIAL_STATES)}")
    print(f"Seeds: {SEEDS}")
    print(f"Total episodes: {len(INITIAL_STATES) * len(SEEDS)}")
    print(f"Base save directory: {BASE_SAVE_DIR}")
    print(f"Save video: {not args.no_video}")
    print(f"Record reconstructions: {args.record_reconstructions}")
    print(f"Device: {DEVICE}")
    print("=" * 80)

    # Run sweep
    print("\n" + "=" * 80)
    print("RUNNING SWEEP")
    print("=" * 80 + "\n")

    episode_count = 0
    max_episodes = args.max_episodes or (len(INITIAL_STATES) * len(SEEDS))

    for seed in SEEDS:
        for initial_state in INITIAL_STATES:
            if episode_count >= max_episodes:
                break

            episode_count += 1
            print(f"\n[Episode {episode_count}/{max_episodes}]")
            print(f"  Seed: {seed}, Initial state: {initial_state}")

            # Set seed
            tools.set_seed_everywhere(seed)

            # Create config
            cfg = create_config(
                checkpoint_path=args.checkpoint,
                seed=seed,
                initial_state=initial_state,
            )

            # Set up agent and environment
            env, agent = setup_dreamer_agent(
                config=cfg,
                checkpoint_path=args.checkpoint,
                record_reconstructions=args.record_reconstructions,
                record_value=True,
            )

            # Create episode-specific save directory
            time_str = datetime.now().strftime("%Y%m%d_%H%M%S")
            state_str = f"cart{initial_state[0]:.1f}_angle{np.rad2deg(initial_state[1]):.0f}"
            episode_dir = BASE_SAVE_DIR / f"seed{seed}" / f"{time_str}_{state_str}"
            activations_dir = episode_dir / 'activations'
            activations_dir.mkdir(parents=True, exist_ok=True)

            # Create recorder
            episode_recorder = DreamerEpisodeDataRecorder(
                save_dir=str(activations_dir),
                record_reconstructions=args.record_reconstructions,
            )

            # Run episode
            episode_stats = run_dreamer_episode_with_recording(
                env=env,
                agent=agent,
                episode_recorder=episode_recorder,
                save_video=not args.no_video,
                episode_dir=str(episode_dir),
                task_name='cartpole_swingup',
                max_steps=cfg.time_limit // cfg.action_repeat,
            )

            # Save episode data with metadata
            metadata = {
                'seed': seed,
                'initial_state': initial_state.tolist(),
                'checkpoint': args.checkpoint,
                'task': cfg.task,
                'episode_reward': episode_stats['reward'],
                'episode_length': episode_stats['length'],
            }
            episode_recorder.save_episode(metadata=metadata)
            print(
                f"  → Reward: {episode_stats['reward']:.2f}, Length: {episode_stats['length']}"
            )
            print(f"  → Saved to: {activations_dir}")

            # Clean up
            try:
                env.close()
            except Exception:
                pass  # Ignore close errors (common with DMC wrappers)

        if episode_count >= max_episodes:
            break

    print("\n" + "=" * 80)
    print("SWEEP COMPLETE!")
    print("=" * 80)
    print(f"Total episodes run: {episode_count}")
    print(f"Results saved to: {BASE_SAVE_DIR}")


if __name__ == '__main__':
    main()
