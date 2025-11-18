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

selected_directions_pca = np.load(
    'sweeps/selected_dirs_pca.npz')['comps_pca_all']
# NOTE: only take the first 5 components for now
selected_directions_pca = selected_directions_pca[:5]

with open('sweeps/selected_dirs_pca_sparse_all.pkl', 'rb') as f:
    selected_dirs_pca_sparse = pickle.load(f)

with open('sweeps/selected_latents_lasso.pkl', 'rb') as f:
    selected_latents_lasso = pickle.load(f)

# SEEDS FOR THE SWEEP
SEEDS = [1, 2, 3, 4, 5]

# Intervention configurations
INTERVENTION_CONFIGS = []

PCA_ABLATION_CONFIGS = []
for seed in SEEDS:
    for hinge_1 in HINGE_1_VALUES:
        for idx, direction in enumerate(selected_directions_pca):
            PCA_ABLATION_CONFIGS.append({
                'type': 'directional_ablation',
                'location': 'encoder_output',
                'direction': direction,
                'hinge_1_value': hinge_1,
                'seed': seed,
                'pca_component_idx': str(idx),
            })

PCA_ABLATION_SCALE_CONFIGS = []
# - scale=0: equivalent to ablation
# - scale=1: no change
# - scale>1: amplify component
# - scale<0: reverse component
scale_direcional_scalings = [
    -0.9, -0.6, -0.3, -0.2, -0.1, 0, 0.1, 0.2, 0.3, 0.6, 0.9, 1.0, 1.2, 1.5, 1.8
]
directions_pca_scaling = selected_directions_pca[:1]
for seed in SEEDS:
    for hinge_1 in HINGE_1_VALUES:
        for idx, direction in enumerate(directions_pca_scaling):
            for scale in scale_direcional_scalings:
                PCA_ABLATION_SCALE_CONFIGS.append({
                    'type': 'directional_scaling_ablation',
                    'location': 'encoder_output',
                    'direction': direction,
                    'scale': scale,
                    'hinge_1_value': hinge_1,
                    'seed': seed,
                    'pca_component_idx': str(idx),
                })

PCA_DIRECTIONAL_ABALATION_SPARSE_CONFIGS = []
selected_dirs_pca_sparse_comp0 = selected_dirs_pca_sparse[0]
idx = 0
eps_fracs_to_sweep = [0.35, 0.4]
for seed in SEEDS:
    for hinge_1 in HINGE_1_VALUES:
        for eps_frac, value in selected_dirs_pca_sparse_comp0.items():
            if eps_frac not in eps_fracs_to_sweep:
                continue
            direction_sparse = value['comp_sparse']
            cosine_similarity = value['cosine']
            n_nonzero = value['n_nonzero']
            PCA_DIRECTIONAL_ABALATION_SPARSE_CONFIGS.append({
                'type': 'directional_ablation_sparse',
                'location': 'encoder_output',
                'direction': direction_sparse,
                'cosine_similarity': cosine_similarity,
                'n_nonzero': n_nonzero,
                'eps_frac': eps_frac,
                'hinge_1_value': hinge_1,
                'seed': seed,
                'pca_component_idx': str(idx),
            })

RANDOM_DIRECTIONS_CONFIGS = []
for seed in SEEDS:
    for hinge_1 in HINGE_1_VALUES:
        RANDOM_DIRECTIONS_CONFIGS.append({
            'type': 'random_direction',
            'location': 'encoder_output',
            'hinge_1_value': hinge_1,
            'seed': seed,
        })

LASSO_ABLATION_CONFIGS = []
for seed in SEEDS:
    for hinge_1 in HINGE_1_VALUES:
        for variable, dict_ in selected_latents_lasso.items():
            for selection_type in ['1se']:
                for selection_threshold in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
                    selected_dims = dict_[selection_type][selection_threshold]
                    LASSO_ABLATION_CONFIGS.append({
                        'type': 'lasso_ablation',
                        'location': 'encoder_output',
                        'selected_dims': selected_dims,
                        'selection_type': selection_type,
                        'selection_threshold': selection_threshold,
                        'variable': variable,
                        'hinge_1_value': hinge_1,
                        'seed': seed,
                    })

PCA_CUMULATIVE_SUFFICIENCY_CONFIGS = []
# Test sufficiency: keep only first k PCA components
k_values_to_test = [1, 2, 3, 4, 5]  # Number of components to keep
for seed in SEEDS:
    for hinge_1 in HINGE_1_VALUES:
        for k in k_values_to_test:
            PCA_CUMULATIVE_SUFFICIENCY_CONFIGS.append({
                'type': 'cumulative_pca_sufficiency',
                'location': 'encoder_output',
                'components':
                    selected_directions_pca[:k],  # Keep first k components
                'k': k,
                'hinge_1_value': hinge_1,
                'seed': seed,
            })

INTERVENTION_CONFIGS = PCA_CUMULATIVE_SUFFICIENCY_CONFIGS
#PCA_DIRECTIONAL_ABALATION_SPARSE_CONFIGS
#PCA_ABLATION_SCALE_CONFIGS  #PCA_ABLATION_CONFIGS + RANDOM_DIRECTIONS_CONFIGS + LASSO_ABLATION_CONFIGS

# Base directory for saving results
BASE_SAVE_DIR = 'logs/latent_interventions_guided_sweep'

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
                             'directional_ablation_sparse'):
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

    elif intervention_type == 'random_direction':
        random_direction = torch.randn(latent_dim,
                                       generator=rng_torch).to(DEVICE)
        random_direction = random_direction / torch.norm(random_direction)
        config['direction'] = random_direction
        patcher.add_directional_ablation(location, direction=random_direction)

    elif intervention_type == 'directional_scaling_ablation':
        scale = config['scale']
        patcher.add_directional_scaling(location,
                                        scale=scale,
                                        direction=direction)

    elif intervention_type == 'cumulative_pca_sufficiency':
        # Keep only first k PCA components (sufficiency test)
        components = config['components']
        k = config['k']

        # Convert components to torch tensor if needed
        if isinstance(components, np.ndarray):
            components = torch.from_numpy(components).float().to(DEVICE)
        else:
            components = components.to(DEVICE)

        # Get SimNorm config from the agent's encoder
        simnorm_dim = 8  # NOTE(R): same as config.yaml

        # Define intervention function: project onto first k components
        def cumulative_sufficiency_intervention(latents, **kwargs):
            """Keep only first k PCA components, zero out all others"""
            # latents shape: (batch, latent_dim)
            # components shape: (k, latent_dim)

            # Project onto first k components and reconstruct
            coeffs = latents @ components.T  # (batch, k)
            latents_reconstructed = coeffs @ components  # (batch, latent_dim)

            # Re-apply SimNorm to maintain the model's expected distribution
            # SimNorm groups dimensions and applies softmax within each group
            shp = latents_reconstructed.shape
            latents_reconstructed = latents_reconstructed.view(
                *shp[:-1], -1, simnorm_dim)
            latents_reconstructed = torch.nn.functional.softmax(
                latents_reconstructed, dim=-1)
            latents_reconstructed = latents_reconstructed.view(*shp)

            return latents_reconstructed

        patcher.add_intervention(location, cumulative_sufficiency_intervention)
        config['components'] = components.tolist()  # Store for metadata

    elif intervention_type == 'lasso_ablation':
        selected_dims = config['selected_dims']
        patcher.add_ablation(location, dims=selected_dims)
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

        #config_id = get_config_id(config)
        #print(f"\n[{i+1}/{len(INTERVENTION_CONFIGS)}] Config: {config_id}")

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
        run_episode_with_recording(env=env,
                                   agent=agent,
                                   episode_recorder=episode_recorder,
                                   planning_recorder=None,
                                   save_video=False,
                                   eval_mode=True,
                                   task=None)

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
