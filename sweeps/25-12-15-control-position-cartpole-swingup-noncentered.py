from logging import lastResort
import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, "/home/hgf_hmgu/hgf_gib4562/tdmpc2/tdmpc2")

from hydra import initialize, compose
from omegaconf import OmegaConf
from utils import setup_agent, run_episode_with_recording
from common.activation_patcher import ActivationPatcher
from common.episode_data_recorder import EpisodeDataRecorder
from common.planning_data_recorder import PlanningDataRecorder
from common.seed import set_seed
from common.parser import parse_cfg
from datetime import datetime
import pickle
import torch
from itertools import product
import os
# ============================================================================
# SWEEP CONFIGURATION
# ============================================================================

# Base configuration
CKPT_PATH = f'ckpts/cartpole-swingup-noncentered.pt'
override_cfg = dict(
    task='cartpole-swingup_custom',
    checkpoint=CKPT_PATH,
    obs='state',
    seed=1,
    compile=False,
    mpc=True,
    multitask=False,
    model_size=5,
    save_video=False,  # Disable video to speed up sweep
    record_planning=False,  # Focus on episode data
)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ============================================================================
# LOAD DATA FOR THE SWEEP
# ============================================================================
path_to_data = '/home/hgf_hmgu/hgf_gib4562/tdmpc2/sweeps/sweep_data/25-12-15-control-position-cartpole-swingup-noncentered'

w_lr_coefs_noncentered_dict = pickle.load(
    open(os.path.join(path_to_data, 'w_lr_coefs_noncentered.pkl'), 'rb'))

# ============================================================================
# GENERATE CONFIGS FOR THE SWEEP
# ============================================================================

# SEEDS FOR THE SWEEP
SEEDS = [0, 1, 2, 3, 4, 5]

LR_CONFIGS = []
for seed in SEEDS:
    for key, value in w_lr_coefs_noncentered_dict.items():

        for magnitude in [-7.5, -5., -2.5, -1, 0, 1, 2.5, 5., 7.5]:
            direction = value['coef']
            LR_CONFIGS.append({
                'type': 'directional_addition',
                'location': 'encoder_output',
                'magnitude': magnitude,
                'seed': seed,
                'direction_str': key,
                'r2': value['r2'],
                'direction': direction,
            })

INTERVENTION_CONFIGS = LR_CONFIGS

# Base directory for saving results
BASE_SAVE_DIR = f'logs/25-12-15-control-position-cartpole-swingup-noncentered_sweep'

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================


def setup_intervention(patcher, config, start_step, end_step):
    """
    Configure the activation patcher based on intervention config.
    
    Args:
        patcher: ActivationPatcher instance
        config: Intervention configuration dict
        latent_dim: Total number of latent dimensions (for random selection)
    
    Returns:
        dims_affected: List of dimension indices that were affected
    """
    intervention_type = config['type']
    location = config['location']

    if intervention_type == 'directional_addition':
        direction = config['direction']
        # Convert direction to torch tensor if it's a numpy array
        if isinstance(direction, np.ndarray):
            direction = torch.from_numpy(direction).float().to(DEVICE)
        else:
            direction = direction.to(DEVICE)

    # Clear any previous interventions
    patcher.clear_interventions()

    if intervention_type == 'directional_addition':
        magnitude = config['magnitude']
        patcher.add_directional_addition(location,
                                         magnitude=magnitude,
                                         direction=direction,
                                         start_step=start_step,
                                         end_step=end_step)
    else:
        raise ValueError(f"Unknown intervention type: {intervention_type}")

    return config


# ============================================================================
# MAIN SWEEP
# ============================================================================


def main():
    print("=" * 80)
    print("LATENT INTERVENTIONS ROBUSTNESS SWEEP")
    print("=" * 80)
    print(f"Task: {override_cfg['task']}")
    print(f"Checkpoint: {override_cfg['checkpoint']}")
    print(f"Total configurations: {len(INTERVENTION_CONFIGS)}")
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

    for i, config in enumerate(INTERVENTION_CONFIGS):

        # Initialize environment and agent
        print("\nInitializing environment and agent...")
        config['hinge_1_value'] = 0.
        cfg.initial_state = {
            'qpos': {
                'hinge_1': 0.,
                'slider': 0.,
            },
            'qvel': {
                'slider': 0.,
                'hinge_1': 0.,
            },
        }
        env, agent = setup_agent(cfg)
        set_seed(config['seed'])

        # Initialize activation patcher
        print("Initializing activation patcher...")
        start_step = 50
        end_step = None
        patcher = ActivationPatcher(
            agent.model,
            modules_to_hook=['encoder_output'],
            renormalize_after_intervention=False,
        )

        config['normalization_method'] = None
        config['start_step'] = start_step
        config['end_step'] = end_step

        # Create episode-specific save directory
        time = datetime.now().strftime("%Y%m%d_%H%M%S")
        episode_dir = Path(BASE_SAVE_DIR) / time
        activations_dir = episode_dir / 'activations'
        planning_dir = episode_dir / 'planning'
        activations_dir.mkdir(parents=True, exist_ok=True)

        # Create episode recorder to save full episode data
        episode_recorder = EpisodeDataRecorder(cfg,
                                               save_dir=str(activations_dir))

        # planning_recorder = PlanningDataRecorder(
        #     save_dir=str(planning_dir),
        #     record_rollouts=True,
        #     record_only_last_iteration=True)
        # Run episode with intervention and recording
        config = setup_intervention(patcher, config, start_step, end_step)
        patcher.enable()

        # Run episode with recording (reuses shared function from analyze_activations.py)
        run_episode_with_recording(
            env=env,
            agent=agent,
            episode_recorder=episode_recorder,
            #planning_recorder=planning_recorder,
            save_video=True,  # change to false for faster evaluation
            eval_mode=True,
            task=None,
            task_name=cfg.task,
            episode_dir=str(episode_dir),
            patcher=patcher)

        # Save episode data with intervention metadata
        # Prepare config for serialization
        config_to_save = config.copy()

        # Convert direction to list so we can serialize it
        if 'direction' in config_to_save:
            config_to_save['direction'] = config_to_save['direction'].tolist()

        # Convert magnitude callable to description string
        if 'magnitude' in config_to_save and callable(
                config_to_save['magnitude']):
            # Replace with description if available, otherwise use a generic string
            config_to_save['magnitude'] = config_to_save.get(
                'magnitude_description', 'callable (not serialized)')

        metadata = {
            'config': config_to_save,
            'checkpoint': str(cfg.checkpoint),
        }
        episode_recorder.save_episode(metadata=metadata)
        #planning_recorder.save_episode(metadata=metadata)
        print(f"  → Saved episode data to: {activations_dir}")
        print(f"  → Saved planning data to: {planning_dir}")

        # Cleanup
        patcher.remove_hooks()

    print("\n" + "=" * 80)
    print("SWEEP COMPLETE!")
    print("=" * 80)
    print(f"Results saved to: {BASE_SAVE_DIR}")


if __name__ == '__main__':
    main()
