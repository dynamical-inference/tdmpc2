"""
Planning data recorder for mechanistic interpretability of TD-MPC2 agents.
Records MPPI planning details during model-based planning.

This recorder captures what the agent "thinks" during planning:
- How action distributions evolve across MPPI iterations
- Which trajectories are selected as elites
- What the world model imagines will happen

Minimal mode (default): Records mean/std/elites (~5 MB per episode)
Full mode (optional): Also records all 512 sampled trajectories (~500 MB per episode)
"""

import numpy as np
import torch
from pathlib import Path
from datetime import datetime
import pickle
import json


class PlanningDataRecorder:
    """
    Records MPPI planning data during agent execution.
    
    Captures:
    - Evolution of action distribution (mean/std) across iterations
    - Elite trajectories selected at each iteration
    - Final action selected
    - Optional: All sampled trajectories (very large)
    
    Minimal recording (default): ~5 MB per episode
    Full recording: ~500 MB per episode
    """

    def __init__(self, save_full_trajectories=False, save_dir=None):
        """
        Initialize planning data recorder.
        
        Args:
            save_full_trajectories: If True, save all 512 sampled trajectories
                                   (very large, ~100x more storage)
            save_dir: Directory to save recordings
        """
        self.save_full_trajectories = save_full_trajectories
        self.save_dir = Path(save_dir) if save_dir else Path(
            'analysis/planning')
        self.save_dir.mkdir(parents=True, exist_ok=True)

        # Storage for current episode
        self.reset_episode()

        # Episode counter
        self.episode_idx = 0

        mode = "full" if save_full_trajectories else "minimal"
        print(
            f"PlanningDataRecorder initialized ({mode} mode). Saving to: {self.save_dir}"
        )

    def reset_episode(self):
        """Reset storage for a new episode."""
        self.planning_steps = []
        self.current_timestep = 0

    def record_planning_step(self,
                             initial_latent,
                             iterations_data,
                             selected_action,
                             all_sampled_actions=None):
        """
        Record one planning step (one call to _plan()).
        
        Args:
            initial_latent: Starting latent state [batch, latent_dim]
            iterations_data: List of dicts, one per MPPI iteration containing:
                - mean_actions: [horizon, action_dim] - mean of action distribution
                - std_actions: [horizon, action_dim] - std of action distribution
                - elite_actions: [num_elites, horizon, action_dim] - top trajectories
                - elite_values: [num_elites] - values of elite trajectories
            selected_action: Final action taken [action_dim]
            all_sampled_actions: Optional [num_samples, horizon, action_dim]
        """
        # Convert to numpy
        step_data = {
            'timestep': self.current_timestep,
            'initial_latent': self._to_numpy(initial_latent),
            'selected_action': self._to_numpy(selected_action),
            'num_iterations': len(iterations_data),
            'iterations': []
        }

        # Store each iteration's data
        for iter_idx, iter_data in enumerate(iterations_data):
            iteration_record = {
                'iteration': iter_idx,
                'mean_actions': self._to_numpy(iter_data['mean_actions']),
                'std_actions': self._to_numpy(iter_data['std_actions']),
                'elite_actions': self._to_numpy(iter_data['elite_actions']),
                'elite_values': self._to_numpy(iter_data['elite_values']),
            }
            step_data['iterations'].append(iteration_record)

        # Optionally store all sampled trajectories
        if self.save_full_trajectories and all_sampled_actions is not None:
            step_data['all_sampled_actions'] = self._to_numpy(
                all_sampled_actions)

        self.planning_steps.append(step_data)
        self.current_timestep += 1

    def _to_numpy(self, tensor):
        """Convert tensor to numpy, handling None and already-numpy cases."""
        if tensor is None:
            return None
        if isinstance(tensor, torch.Tensor):
            return tensor.detach().cpu().numpy()
        return tensor

    def save_episode(self, metadata=None):
        """
        Save the current episode's planning data to disk.
        
        Args:
            metadata: Optional dict with additional metadata
        """
        # Prepare episode data
        episode_dict = {
            'planning_steps': self.planning_steps,
            'num_steps': len(self.planning_steps),
        }

        # Add metadata
        if metadata is None:
            metadata = {}

        metadata.update({
            'episode_idx': self.episode_idx,
            'num_planning_steps': len(self.planning_steps),
            'save_full_trajectories': self.save_full_trajectories,
            'timestamp': datetime.now().isoformat(),
        })

        # Save to file
        filename = f"planning_episode_{self.episode_idx:04d}.pkl"
        filepath = self.save_dir / filename

        save_data = {
            'metadata': metadata,
            'data': episode_dict,
        }

        with open(filepath, 'wb') as f:
            pickle.dump(save_data, f)

        # Also save metadata as JSON
        metadata_file = self.save_dir / f"planning_episode_{self.episode_idx:04d}_metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)

        # Estimate size
        file_size_mb = filepath.stat().st_size / (1024 * 1024)
        print(
            f"Saved planning data episode {self.episode_idx}: {len(self.planning_steps)} steps, "
            f"{file_size_mb:.2f} MB -> {filepath}")

        self.episode_idx += 1
        self.reset_episode()

    def get_current_episode_data(self):
        """Return the current episode data (for in-memory analysis)."""
        return {
            'planning_steps': self.planning_steps,
            'num_steps': len(self.planning_steps),
        }
