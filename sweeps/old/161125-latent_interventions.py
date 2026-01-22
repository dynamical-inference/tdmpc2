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

negative_range = np.linspace(-3.14, -1.5, 5)  # 5 episodes in the negative range
positive_range = np.linspace(1.5, 3.14, 5)  # 4 episodes in the positive range
HINGE_1_VALUES = np.concatenate([negative_range, positive_range]).round(2)

# SEEDS FOR THE SWEEP
SEEDS = [1, 2, 3, 4, 5]

# Intervention configurations
# Each config is a dict with: type, location, params
INTERVENTION_CONFIGS = []

# --- ABLATION SWEEP: Vary number of dimensions ---
ABLATION_NUM_DIMS = [0, 10, 50, 100, 200, 300, 400]
for num_dims in ABLATION_NUM_DIMS:
    for seed in SEEDS:
        for hinge_1 in HINGE_1_VALUES:
            INTERVENTION_CONFIGS.append({
                'type': 'ablation',
                'location': 'encoder_output',
                'num_dims': num_dims,
                'hinge_1_value': hinge_1,
                'seed': seed,
            })

# --- NOISE SWEEP: Vary scale and number of dimensions ---
NOISE_SCALES = [0.1, 0.5, 1.0]
NOISE_NUM_DIMS = [0, 10, 50, 100, 200, 300, 400]

for scale in NOISE_SCALES:
    for num_dims in NOISE_NUM_DIMS:
        for seed in SEEDS:
            for hinge_1 in HINGE_1_VALUES:
                INTERVENTION_CONFIGS.append({
                    'type': 'noise',
                    'location': 'encoder_output',
                    'num_dims': num_dims,
                    'scale': scale,
                    'hinge_1_value': hinge_1,
                    'seed': seed,
                })

# Base directory for saving results
BASE_SAVE_DIR = 'logs/latent_interventions_sweep'

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
    num_dims = config['num_dims']
    seed = config['seed']

    # Clear any previous interventions
    patcher.clear_interventions()

    # Select random dimensions using seed
    rng = np.random.RandomState(seed)
    dims_affected = rng.choice(latent_dim,
                               size=min(num_dims, latent_dim),
                               replace=False).tolist()

    if intervention_type == 'ablation':
        # Zero out specific dimensions
        patcher.add_ablation(location, dims=dims_affected)

    elif intervention_type == 'noise':
        # Add Gaussian noise to specific dimensions
        scale = config['scale']
        patcher.add_noise(location, dims=dims_affected, scale=scale)

    else:
        raise ValueError(f"Unknown intervention type: {intervention_type}")

    return dims_affected


def get_config_id(config):
    """Generate a unique identifier for an intervention config."""
    hinge1 = config['hinge_1_value']
    if config['type'] == 'ablation':
        return f"ablationdims{config['num_dims']}-hinge1_{hinge1}-seed{config['seed']}"
    elif config['type'] == 'noise':
        return f"noisedims{config['num_dims']}-scale{config['scale']}-hinge1_{hinge1}-seed{config['seed']}"
    else:
        raise ValueError(f"Unknown intervention type: {config['type']}")


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

    # Storage for all results
    all_results = []

    # Run sweep
    print("\n" + "=" * 80)
    print("RUNNING SWEEP")
    print("=" * 80 + "\n")

    for i, config in enumerate(INTERVENTION_CONFIGS):

        if config['hinge_1_value'] == -3.14:
            continue

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

        config_id = get_config_id(config)
        print(f"\n[{i+1}/{len(INTERVENTION_CONFIGS)}] Config: {config_id}")

        # Create episode-specific save directory
        time = datetime.now().strftime("%Y%m%d_%H%M%S")
        episode_dir = Path(BASE_SAVE_DIR) / time
        activations_dir = episode_dir / 'activations'
        activations_dir.mkdir(parents=True, exist_ok=True)

        # Create episode recorder to save full episode data
        episode_recorder = EpisodeDataRecorder(cfg,
                                               save_dir=str(activations_dir))

        # Run episode with intervention and recording
        dims_affected = setup_intervention(patcher, config, cfg.latent_dim)
        patcher.enable()

        # Run episode with recording (reuses shared function from analyze_activations.py)
        run_episode_with_recording(env=env,
                                   agent=agent,
                                   episode_recorder=episode_recorder,
                                   planning_recorder=None,
                                   save_video=False,
                                   eval_mode=True,
                                   task=None)

        # Save episode data with intervention metadata
        # change config direction to list so we can serialize it
        config['direction'] = config['direction'].tolist()
        metadata = {
            'config_id': config_id,
            'config': config,
            'dims_affected': dims_affected,
            'checkpoint': str(cfg.checkpoint),
            'task': cfg.task,
            'latent_dim': cfg.latent_dim,
        }
        episode_recorder.save_episode(metadata=metadata)
        # Save planning data if recording
        #if planning_recorder is not None:
        #    planning_recorder.save_episode(metadata=metadata)
        print(f"  → Saved episode data to: {activations_dir}")

        # Store results
        result = {
            'config_id': config_id,
            'config': config,
            'dims_affected': dims_affected,
            'save_dir': str(episode_dir),
        }
        all_results.append(result)

        # Cleanup
        patcher.remove_hooks()

    print("\n" + "=" * 80)
    print("SWEEP COMPLETE!")
    print("=" * 80)
    print(f"Results saved to: {BASE_SAVE_DIR}")


if __name__ == '__main__':
    main()
