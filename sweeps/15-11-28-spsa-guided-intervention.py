import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, "/home/hgf_hmgu/hgf_gib4562/tdmpc2/tdmpc2")

from hydra import initialize, compose
from omegaconf import OmegaConf
from utils import setup_agent, run_episode_with_recording
from common.activation_patcher import ActivationPatcher
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
CKPT_PATH = '/home/hgf_hmgu/hgf_gib4562/tdmpc2/ckpts/cartpole-swingup-3.pt'
override_cfg = dict(
    task='cartpole-swingup',
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

negative_range = np.linspace(-3.14, -1.5, 5)  # 5 episodes in the negative range
positive_range = np.linspace(1.5, 3.14, 5)  # 4 episodes in the positive range
HINGE_1_VALUES = np.concatenate([negative_range, positive_range]).round(2)

# Load gradient data
with open(
        '/home/hgf_hmgu/hgf_gib4562/tdmpc2/sweeps/gradient_position_3seeds.pkl',
        'rb') as f:
    gradients_position_3seeds = pickle.load(f)

with open(
        '/home/hgf_hmgu/hgf_gib4562/tdmpc2/sweeps/gradient_position_3seeds_h3.pkl',
        'rb') as f:
    gradients_position_3seeds_h3 = pickle.load(f)

with open(
        '/home/hgf_hmgu/hgf_gib4562/tdmpc2/sweeps/gradient_position_3seeds_h10.pkl',
        'rb') as f:
    gradients_position_3seeds_h10 = pickle.load(f)

# SEEDS FOR THE SWEEP
SEEDS = [1, 2, 3, 4, 5]

# Intervention configurations
INTERVENTION_CONFIGS = []

AMP_GRADIENT_CONFIGS = []
#for seed in SEEDS:
for gradient in gradients_position_3seeds_h3:
    for magnitude in [-7.5]:
        AMP_GRADIENT_CONFIGS.append({
            'type': 'directional_addition_gradient',
            'location': 'encoder_output',
            'direction': gradient.gradient,
            'magnitude': magnitude,
            'hinge_1_value': -3.14,
            'seed': 1,
        })

INTERVENTION_CONFIGS = AMP_GRADIENT_CONFIGS
#AMP_GRADIENT_CONFIGS
#AMP_DIRECTION_CONFIGS
#PCA_CUMULATIVE_SUFFICIENCY_CONFIGS
#PCA_DIRECTIONAL_ABALATION_SPARSE_CONFIGS
#PCA_ABLATION_SCALE_CONFIGS  #PCA_ABLATION_CONFIGS + RANDOM_DIRECTIONS_CONFIGS + LASSO_ABLATION_CONFIGS

# Base directory for saving results
BASE_SAVE_DIR = 'logs/spsa-guided-intervention'

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================


def setup_intervention(patcher, config, latent_dim):
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

    if intervention_type in ('directional_ablation',
                             'directional_scaling_ablation',
                             'directional_ablation_sparse',
                             'directional_addition',
                             'directional_addition_gradient'):
        direction = config['direction']
        # Convert direction to torch tensor if it's a numpy array
        if isinstance(direction, np.ndarray):
            direction = torch.from_numpy(direction).float().to(DEVICE)
        else:
            direction = direction.to(DEVICE)

    rng_torch = torch.Generator().manual_seed(config['seed'])

    # Clear any previous interventions
    patcher.clear_interventions()

    if intervention_type in [
            'directional_ablation', 'directional_ablation_sparse'
    ]:
        patcher.add_directional_ablation(location, direction=direction)

    elif intervention_type in [
            'directional_addition', 'directional_addition_gradient'
    ]:
        magnitude = config['magnitude']
        patcher.add_directional_addition(location,
                                         magnitude=magnitude,
                                         direction=direction)
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

    for i, config in enumerate(INTERVENTION_CONFIGS):

        # if config['hinge_1_value'] != -3.14:
        #     continue

        # Initialize environment and agent
        print("\nInitializing environment and agent...")
        hinge_1 = config['hinge_1_value']
        cfg.initial_state = {
            'qpos': {
                'hinge_1': float(hinge_1),
            },
        }
        env, agent = setup_agent(cfg)

        # Initialize activation patcher
        print("Initializing activation patcher...")
        patcher = ActivationPatcher(agent.model,
                                    modules_to_hook=['encoder_output'])

        # Create episode-specific save directory
        time = datetime.now().strftime("%Y%m%d_%H%M%S")
        episode_dir = Path(BASE_SAVE_DIR) / time
        activations_dir = episode_dir / 'activations'
        activations_dir.mkdir(parents=True, exist_ok=True)

        # Create episode recorder to save full episode data
        episode_recorder = EpisodeDataRecorder(cfg,
                                               save_dir=str(activations_dir))

        # Run episode with intervention and recording
        config = setup_intervention(patcher, config, cfg.latent_dim)
        patcher.enable()

        # Run episode with recording (reuses shared function from analyze_activations.py)
        run_episode_with_recording(
            env=env,
            agent=agent,
            episode_recorder=episode_recorder,
            planning_recorder=None,
            save_video=True,  # change to false for faster evaluation
            eval_mode=True,
            task=None,
            task_name=cfg.task,
            episode_dir=str(episode_dir))

        # Save episode data with intervention metadata
        # change config direction to list so we can serialize it
        if 'direction' in config:
            config['direction'] = config['direction'].tolist()

        metadata = {
            'config': config,
            'checkpoint': str(cfg.checkpoint),
            'task': cfg.task,
            'latent_dim': cfg.latent_dim,
        }
        episode_recorder.save_episode(metadata=metadata)
        print(f"  → Saved episode data to: {activations_dir}")

        # Cleanup
        patcher.remove_hooks()

    print("\n" + "=" * 80)
    print("SWEEP COMPLETE!")
    print("=" * 80)
    print(f"Results saved to: {BASE_SAVE_DIR}")


if __name__ == '__main__':
    main()
