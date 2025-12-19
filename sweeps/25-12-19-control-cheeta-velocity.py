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
import pickle
from common.activation_patcher import ActivationPatcher

# ============================================================================
# LOAD DATA
# ============================================================================

path_to_w_velocity_probes_coef = '/home/hgf_hmgu/hgf_gib4562/tdmpc2/sweeps/sweep_data/25-12-19-control-cheetah-velocity-probe/w_velocity_probes_coef.pkl'
with open(path_to_w_velocity_probes_coef, 'rb') as f:
    w_velocity_probes_coef_dict = pickle.load(f)

# ============================================================================
# SWEEP CONFIGURATION
# ============================================================================

# Base configuration
override_cfg = dict(
    task='cheetah-run',
    checkpoint='ckpts/cheetah-run-3.pt',
    obs='state',
    seed=1,
    compile=False,
    mpc=True,
    save_video=True,
    multitask=False,
    record_planning=False,
)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Base directory for saving results
BASE_SAVE_DIR = f'logs/25-12-19-control-cheetah-velocity-probe'

# SEEDS FOR THE SWEEP
SEEDS = [0, 1, 2]

LR_CONFIGS_LAGGED_DELTA = []
for seed in SEEDS:
    for start_step in [0, 75]:
        for key, value in w_velocity_probes_coef_dict.items():
            for magnitude in [-7.5, -5., -2.5, -1, 0, 1, 2.5, 5., 7.5]:
                LR_CONFIGS_LAGGED_DELTA.append({
                    'type': 'directional_addition',
                    'location': 'encoder_output',
                    'magnitude': magnitude,
                    'seed': seed,
                    'direction_str': key,
                    'r2': value['r2'] if 'r2' in value else None,
                    'start_step': start_step,
                    'direction': value['coef'],
                })


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
    print("INITIAL CONDITIONS SEEDS SWEEP")
    print("=" * 80)
    print(f"Seeds: {SEEDS}")
    print(f"Base save directory: {BASE_SAVE_DIR}")
    print("=" * 80)

    episode_count = 0
    for config in LR_CONFIGS_LAGGED_DELTA:
        episode_count += 1

        # Initialize Hydra config
        with initialize(config_path="../tdmpc2", version_base=None):
            cfg = compose(config_name="config")
            OmegaConf.set_struct(cfg, False)
            cfg = OmegaConf.merge(cfg, override_cfg)

        config['task_name'] = 'cheetah-run'

        # Parse config
        cfg = parse_cfg(cfg)

        # Set seed for this episode
        seed = config['seed']
        set_seed(seed)

        # Initialize environment and agent
        env, agent = setup_agent(cfg)

        patcher = ActivationPatcher(
            agent.model,
            modules_to_hook=['encoder_output'],
            #renormalize_after_intervention=False,
        )
        patcher.enable()

        # Create episode-specific save directory
        time = datetime.now().strftime("%Y%m%d_%H%M%S")
        episode_dir = Path(BASE_SAVE_DIR) / time
        activations_dir = episode_dir / 'activations'
        activations_dir.mkdir(parents=True, exist_ok=True)

        # Create episode recorder to save full episode data
        episode_recorder = EpisodeDataRecorder(cfg,
                                               save_dir=str(activations_dir))

        config = setup_intervention(patcher, config, config['start_step'], None)

        # Run episode without any interventions
        run_episode_with_recording(env=env,
                                   agent=agent,
                                   episode_recorder=episode_recorder,
                                   planning_recorder=None,
                                   save_video=cfg.save_video,
                                   eval_mode=True,
                                   task=None,
                                   task_name=config['task_name'],
                                   episode_dir=str(episode_dir),
                                   patcher=patcher)

        # Save episode data with intervention metadata
        # Prepare config for serialization
        config_to_save = config.copy()

        # Convert direction to list so we can serialize it
        if 'direction' in config_to_save:
            config_to_save['direction'] = config_to_save['direction'].tolist()

        # Save episode data with metadata
        metadata = {
            'config': config_to_save,
            'seed': seed,
            'task_name': config['task_name'],
            'checkpoint': str(cfg.checkpoint),
        }
        episode_recorder.save_episode(metadata=metadata)
        print(f"  → Saved episode data to: {activations_dir}")

        # Cleanup
        patcher.remove_hooks()

    print("\n" + "=" * 80)
    print("SWEEP COMPLETE!")
    print("=" * 80)
    print(f"Total episodes run: {episode_count}")
    print(f"Results saved to: {BASE_SAVE_DIR}")


if __name__ == '__main__':
    main()
