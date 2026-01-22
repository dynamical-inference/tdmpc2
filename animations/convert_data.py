"""
Convert episode_data.pkl to JSON for the web visualization.
Run this script to generate episode_data.json from your pickle file.
"""

import pickle
import json
import numpy as np


def convert_episode_to_json(pkl_path, output_path):
    """Convert pickle episode data to JSON for web visualization."""

    # Load pickle data
    with open(pkl_path, 'rb') as f:
        episode_data = pickle.load(f)

    # Extract data
    observations = np.array(episode_data['data']['observations'])  # (T, 5)
    latent_states = np.concatenate(episode_data['data']['latent_states'],
                                   axis=0)  # (T, 512)
    actions = np.array(episode_data['data']['actions']).reshape(-1, 1)  # (T, 1)
    rewards = np.array(episode_data['data']['rewards']).reshape(-1, 1)  # (T, 1)

    n_timesteps = observations.shape[0]
    n_neurons = latent_states.shape[1]

    print(f"Observations shape: {observations.shape}")
    print(f"Latent states shape: {latent_states.shape}")
    print(f"Actions shape: {actions.shape}")
    print(f"Rewards shape: {rewards.shape}")

    # Normalize latent states to [0, 1] for visualization
    z_min = latent_states.min(axis=0, keepdims=True)
    z_max = latent_states.max(axis=0, keepdims=True)
    z_range = z_max - z_min
    z_range[z_range == 0] = 1  # Avoid division by zero
    activations_normalized = (latent_states - z_min) / z_range

    # Convert observations to individual variables
    # obs = [position, cos(angle), sin(angle), cart_velocity, pole_angular_velocity]
    position = observations[:, 0]
    cos_angle = observations[:, 1]
    sin_angle = observations[:, 2]
    cart_velocity = observations[:, 3]
    angular_velocity = observations[:, 4]

    # Convert angle from sin/cos to degrees
    angle_rad = np.arctan2(sin_angle, cos_angle)
    angle_deg = np.degrees(angle_rad)

    # Build JSON structure
    data = {
        'metadata': {
            'n_timesteps': int(n_timesteps),
            'n_neurons': int(n_neurons),
        },
        'activations': activations_normalized.tolist(),
        'observations': {
            'position': position.tolist(),
            'velocity': cart_velocity.tolist(),
            'angle': angle_deg.tolist(),
            'angular_velocity': angular_velocity.tolist(),
        },
        'actions': actions.flatten().tolist(),
        'rewards': rewards.flatten().tolist(),
    }

    # Save to JSON
    with open(output_path, 'w') as f:
        json.dump(data, f)

    import os
    file_size = os.path.getsize(output_path) / 1024 / 1024
    print(f"\nSaved to {output_path} ({file_size:.2f} MB)")
    print(f"Timesteps: {n_timesteps}, Neurons: {n_neurons}")

    return data


if __name__ == '__main__':
    # Convert the data
    pkl_path = '20260104_185854/activations/episode_data.pkl'
    output_path = '20260104_185854/episode_data.json'

    convert_episode_to_json(pkl_path, output_path)
