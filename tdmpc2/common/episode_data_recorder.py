"""
Episode data recorder for mechanistic interpretability of TD-MPC2 agents.
Records latent states, actions, observations, and rewards during episode execution
and saves them to disk for offline analysis.

This recorder captures data explicitly via a wrapper (no PyTorch hooks).
For real-time activation patching/modification, see activation_patcher.py.
"""

import numpy as np
import torch
from pathlib import Path
from datetime import datetime
import pickle
import json


class EpisodeDataRecorder:
    """
    Records episode data for mechanistic interpretability analysis.
    
    Captures latent states, actions, observations, and rewards during episodes
    and saves them to disk as .pkl files for offline analysis.
    
    Note: Uses explicit capture via wrapper (no PyTorch hooks).
    If using with ActivationPatcher, this will automatically capture the
    MODIFIED activations after patches are applied.
    
    Level 1: Records latent states, actions, observations, and rewards.
    """

    def __init__(self, cfg, save_dir=None):
        """
		Initialize the activation recorder.
		
		Args:
			cfg: Configuration object
			save_dir: Directory to save recordings (default: analysis/activations)
		"""
        self.cfg = cfg

        if save_dir is not None:
            self.save_dir = Path(save_dir) if save_dir else Path(
                'analysis/activations')
            self.save_dir.mkdir(parents=True, exist_ok=True)
            print(
                f"EpisodeDataRecorder initialized. Saving to: {self.save_dir}")
        else:
            print(
                "EpisodeDataRecorder initialized. No save directory provided.")
        # Storage for current episode
        self.reset_episode()

        # Episode counter
        self.episode_idx = 0

    def reset_episode(self):
        """Reset storage for a new episode."""
        self.episode_data = {
            'observations': [],
            'actions': [],
            'rewards': [],
            'dones': [],
            'latent_states': [],
            'timesteps': [],
        }
        self.current_timestep = 0

    def record_step(self, obs, action, reward, done, latent_state):
        """
		Record data for a single timestep.
		
		Args:
			obs: Observation (tensor or dict)
			action: Action taken (tensor)
			reward: Reward received (float or tensor)
			done: Whether episode is done (bool)
			latent_state: Latent state from encoder (tensor)
		"""
        # Convert to numpy for storage
        if isinstance(obs, dict):
            obs_np = {
                k: v.cpu().numpy() if torch.is_tensor(v) else v
                for k, v in obs.items()
            }
        else:
            obs_np = obs.cpu().numpy() if torch.is_tensor(obs) else obs

        action_np = action.cpu().numpy() if torch.is_tensor(action) else action
        reward_np = float(reward)
        latent_np = latent_state.cpu().numpy() if torch.is_tensor(
            latent_state) else latent_state

        # Store data
        self.episode_data['observations'].append(obs_np)
        self.episode_data['actions'].append(action_np)
        self.episode_data['rewards'].append(reward_np)
        self.episode_data['dones'].append(done)
        self.episode_data['latent_states'].append(latent_np)
        self.episode_data['timesteps'].append(self.current_timestep)

        self.current_timestep += 1

    def save_episode(self, metadata=None):
        """
		Save the current episode data to disk.
		
		Args:
			metadata: Optional dict with additional metadata (task, reward, etc.)
		"""
        # Convert lists to arrays for efficient storage
        episode_dict = {
            'observations':
                self.episode_data['observations'],  # Keep as list if dict
            'actions': np.array(self.episode_data['actions']),
            'rewards': np.array(self.episode_data['rewards']),
            'dones': np.array(self.episode_data['dones']),
            'latent_states': np.array(self.episode_data['latent_states']),
            'timesteps': np.array(self.episode_data['timesteps']),
        }

        # Add metadata
        if metadata is None:
            metadata = {}

        metadata.update({
            'episode_idx': self.episode_idx,
            'episode_length': self.current_timestep,
            'total_reward': float(np.sum(episode_dict['rewards'])),
            'timestamp': datetime.now().isoformat(),
            'task': self.cfg.task,
            'obs_type': self.cfg.obs,
        })

        # Save to file
        filename = f"episode_data.pkl"
        filepath = self.save_dir / filename

        save_data = {
            'metadata': metadata,
            'data': episode_dict,
        }

        with open(filepath, 'wb') as f:
            pickle.dump(save_data, f)

        # Also save metadata as JSON for easy inspection
        metadata_file = self.save_dir / f"episode_metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)

        print(
            f"Saved episode {self.episode_idx}: {self.current_timestep} steps, "
            f"reward: {metadata['total_reward']:.2f} -> {filepath}")

        self.episode_idx += 1
        self.reset_episode()

    def get_current_episode_data(self):
        """Return the current episode data (useful for in-memory analysis)."""
        return {
            'observations': self.episode_data['observations'],
            'actions': np.array(self.episode_data['actions']),
            'rewards': np.array(self.episode_data['rewards']),
            'dones': np.array(self.episode_data['dones']),
            'latent_states': np.array(self.episode_data['latent_states']),
            'timesteps': np.array(self.episode_data['timesteps']),
        }
