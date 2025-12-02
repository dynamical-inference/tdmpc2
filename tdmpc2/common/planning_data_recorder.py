"""
Planning data recorder for mechanistic interpretability of TD-MPC2 agents.
Records MPPI planning details during model-based planning.

This recorder captures what the agent "thinks" during planning:
- How action distributions evolve across MPPI iterations
- Which trajectories are selected as elites
- Elite trajectory scores (softmax weights)
- Optional: Dynamics model rollouts (predicted latent states and rewards)
"""

import numpy as np
import torch
from pathlib import Path
from datetime import datetime
import pickle


class PlanningDataRecorder:
    """
    Records MPPI planning data during agent execution.
    
    Captures:
    - Evolution of action distribution (mean/std) across iterations
    - Elite trajectories selected at each iteration
    - Elite scores (softmax weights)
    - Final action selected
    - Optional: Dynamics model rollouts (latent states and rewards)
    - Optional: Only record last iteration per step (saves memory/disk space)
    """

    def __init__(self,
                 save_dir,
                 record_rollouts=False,
                 record_only_last_iteration=False):
        """
        Initialize planning data recorder.
        
        Args:
            save_dir: Directory to save recordings
            record_rollouts: If True, record dynamics model rollouts (latents and rewards)
            record_only_last_iteration: If True, only record the last MPPI iteration per step
        """
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.record_rollouts = record_rollouts
        self.record_only_last_iteration = record_only_last_iteration

        # Storage for current episode
        self.reset_episode()

        print(f"PlanningDataRecorder initialized. Saving to: {self.save_dir}")
        if record_only_last_iteration:
            print("  - Recording only last iteration per planning step")

    def reset_episode(self):
        """Reset storage for a new episode."""
        self.planning_steps = []

    def record_planning_step(self,
                             iterations_data,
                             final_actions=None,
                             final_action=None,
                             random_idx=None):
        """
        Record one planning step (one call to _plan()).
        
        Args:
            iterations_data: List of dicts, one per MPPI iteration containing:
                - mean_actions: [horizon, action_dim] - mean of action distribution
                - std_actions: [horizon, action_dim] - std of action distribution
                - elite_actions: [horizon, num_elites, action_dim] - top trajectories
                - elite_values: [num_elites, 1] - values of elite trajectories
                - score: [num_elites, 1] - softmax weights for elites
                - rollout_elite_latents: Optional [horizon+1, num_elites, latent_dim]
                - rollout_elite_rewards: Optional [horizon, num_elites, 1]
            final_actions: All final actions [horizon, action_dim] (before adding noise)
            final_action: Final action taken [action_dim] (after adding noise in non-eval mode)
        """
        # Convert to numpy
        step_data = {
            'final_action': self._to_numpy(final_action),
            'final_actions': self._to_numpy(final_actions),
            'random_idx': self._to_numpy(random_idx),
            'iterations': []
        }

        # Store each iteration's data (or just the last one if configured)
        iterations_to_record = [
            iterations_data[-1]
        ] if self.record_only_last_iteration else iterations_data

        for iter_idx, iter_data in enumerate(iterations_to_record):
            # Use actual iteration index if recording all, otherwise it's the last iteration
            actual_iter_idx = len(
                iterations_data
            ) - 1 if self.record_only_last_iteration else iter_idx

            iteration_record = {
                'iteration': actual_iter_idx,
                'mean_actions': self._to_numpy(iter_data['mean_actions']),
                'std_actions': self._to_numpy(iter_data['std_actions']),
                'elite_actions': self._to_numpy(iter_data['elite_actions']),
                'elite_values': self._to_numpy(iter_data['elite_values']),
                'score': self._to_numpy(iter_data.get('score')),
            }

            # Add rollout data if present
            if 'rollout_elite_latents' in iter_data:
                iteration_record['rollout_elite_latents'] = self._to_numpy(
                    iter_data['rollout_elite_latents'])
            if 'rollout_elite_rewards' in iter_data:
                iteration_record['rollout_elite_rewards'] = self._to_numpy(
                    iter_data['rollout_elite_rewards'])

            step_data['iterations'].append(iteration_record)

        self.planning_steps.append(step_data)

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
        if metadata is None:
            metadata = {}

        # Build complete save data
        save_data = {
            'timestamp': datetime.now().isoformat(),
            'planning_steps': self.planning_steps,
            **metadata  # Merge any additional metadata
        }

        # Save to file
        filename = f"planning.pkl"
        filepath = self.save_dir / filename

        with open(filepath, 'wb') as f:
            pickle.dump(save_data, f)

        file_size_mb = filepath.stat().st_size / (1024 * 1024)
        print(f"Saved planning: "
              f"{len(self.planning_steps)} steps, {file_size_mb:.2f} MB")

        self.reset_episode()
