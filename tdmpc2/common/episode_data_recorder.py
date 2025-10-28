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
        self.save_dir = Path(save_dir) if save_dir else Path(
            'analysis/activations')
        self.save_dir.mkdir(parents=True, exist_ok=True)

        # Storage for current episode
        self.reset_episode()

        # Episode counter
        self.episode_idx = 0

        print(f"EpisodeDataRecorder initialized. Saving to: {self.save_dir}")

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
        filename = f"episode_{self.episode_idx:04d}.pkl"
        filepath = self.save_dir / filename

        save_data = {
            'metadata': metadata,
            'data': episode_dict,
        }

        with open(filepath, 'wb') as f:
            pickle.dump(save_data, f)

        # Also save metadata as JSON for easy inspection
        metadata_file = self.save_dir / f"episode_{self.episode_idx:04d}_metadata.json"
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


class EpisodeDataRecorderWrapper:
    """
    Wrapper around TD-MPC2 agent that captures episode data during act().
    
    Explicitly calls model.encode() and captures the latent state for recording.
    Compatible with ActivationPatcher: if patches are active, this will capture
    the MODIFIED latent state after patches have been applied.
    """

    def __init__(self, agent, recorder):
        """
		Initialize wrapper.
		
		Args:
			agent: TDMPC2 agent
			recorder: ActivationRecorder instance
		"""
        self.agent = agent
        self.recorder = recorder
        self._last_latent = None

    #NOTE(Rodrigo): This is the original act() method.

    # @torch.no_grad()
    # def act(self, obs, t0=False, eval_mode=False, task=None):
    # 	"""
    # 	Select an action by planning in the latent space of the world model.

    # 	Args:
    # 		obs (torch.Tensor): Observation from the environment.
    # 		t0 (bool): Whether this is the first observation in the episode.
    # 		eval_mode (bool): Whether to use the mean of the action distribution.
    # 		task (int): Task index (only used for multi-task experiments).

    # 	Returns:
    # 		torch.Tensor: Action to take in the environment.
    # 	"""
    # 	obs = obs.to(self.device, non_blocking=True).unsqueeze(0)
    # 	if task is not None:
    # 		task = torch.tensor([task], device=self.device)
    # 	if self.cfg.mpc:
    # 		return self.plan(obs, t0=t0, eval_mode=eval_mode, task=task).cpu()
    # 	z = self.model.encode(obs, task)
    # 	action, info = self.model.pi(z, task)
    # 	if eval_mode:
    # 		action = info["mean"]
    # 	return action[0].cpu()

    def act(self, obs, t0=False, eval_mode=False, task=None):
        """
		Act and record latent state.
		
		Args:
			obs: Observation
			t0: Whether first timestep
			eval_mode: Whether in eval mode
			task: Task ID (for multi-task)
			
		Returns:
			action: Action to take
		"""
        # Prepare observation
        obs_tensor = obs.to(self.agent.device, non_blocking=True).unsqueeze(0)
        if task is not None:
            task_tensor = torch.tensor([task], device=self.agent.device)
        else:
            task_tensor = None

        # Encode to get latent state
        latent_state = self.agent.model.encode(obs_tensor, task_tensor)

        # Store latent state for recording
        self._last_latent = latent_state.detach()

        # Get action (either via planning or policy)
        if self.agent.cfg.mpc:
            action = self.agent.plan(obs_tensor,
                                     t0=t0,
                                     eval_mode=eval_mode,
                                     task=task_tensor)
        else:
            action, _ = self.agent.model.pi(latent_state, task_tensor)
            if eval_mode:
                _, info = self.agent.model.pi(latent_state, task_tensor)
                action = info["mean"]

        return action.cpu()

    def get_last_latent(self):
        """Return the latent state from the last act() call."""
        return self._last_latent

    def __getattr__(self, name):
        """Delegate all other attributes to the wrapped agent."""
        return getattr(self.agent, name)


def load_episode(filepath):
    """
	Load a saved episode from disk.
	
	Args:
		filepath: Path to episode pickle file
		
	Returns:
		dict with 'metadata' and 'data' keys
	"""
    with open(filepath, 'rb') as f:
        episode_data = pickle.load(f)
    return episode_data


def load_all_episodes(save_dir):
    """
	Load all episodes from a directory.
	
	Args:
		save_dir: Directory containing episode files
		
	Returns:
		list of episode dicts
	"""
    save_dir = Path(save_dir)
    episode_files = sorted(save_dir.glob("episode_*.pkl"))

    episodes = []
    for filepath in episode_files:
        episodes.append(load_episode(filepath))

    print(f"Loaded {len(episodes)} episodes from {save_dir}")
    return episodes
