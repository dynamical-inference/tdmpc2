import os
import sys
import numpy as np

sys.path.insert(0, "/home/hgf_hmgu/hgf_gib4562/tdmpc2/tdmpc2")

from hydra import initialize, compose
from omegaconf import OmegaConf
from analyze_activations import analyze_activations

# Number of episodes to run
num_episodes = 10

# Generate hinge_1 values from [-3.14, -1.5] U [1.5, 3.14]
# Split episodes between the two ranges
episodes_per_range = num_episodes // 2

# Generate values from the two ranges
negative_range = np.linspace(-3.14, -1.5, episodes_per_range)
positive_range = np.linspace(1.5, 3.14, num_episodes - episodes_per_range)

# Combine the ranges
hinge_1_values = np.concatenate([negative_range, positive_range])

override_cfg = dict(
    task='cartpole-swingup',
    checkpoint=
    '/home/hgf_hmgu/hgf_gib4562/tdmpc2/tdmpc2/ckpts/cartpole-swingup-3.pt',
    obs='state',
    seed=1,
    compile=False,
    mpc=True,
    multitask=False,
    model_size=5,
    save_video=True,
    record_planning=True,
    #base_save_dir='analysis/initial_conditions',
)

with initialize(config_path="../tdmpc2", version_base=None):
    # config_name is the base name of the YAML file (without .yaml extension)
    cfg = compose(config_name="config")

    # Set struct mode to False to allow adding new keys not in base config
    OmegaConf.set_struct(cfg, False)

    # Merge overrides onto the base config
    cfg = OmegaConf.merge(cfg, override_cfg)

# Run analysis for each hinge_1 value
for i, hinge_1 in enumerate(hinge_1_values):
    print(f"\n{'='*60}")
    print(f"Running episode {i+1}/{num_episodes} with hinge_1={hinge_1:.4f}")
    print(f"{'='*60}\n")

    # Update the hinge_1 value in the config
    cfg.initial_state = {
        'qpos': {
            'hinge_1': float(hinge_1),  # Will be overridden in the loop
        },
    }

    cfg.save_dir = f'analysis/initial_conditions/hinge_1_{hinge_1:.4f}'
    # Run the analysis
    analyze_activations(cfg)
