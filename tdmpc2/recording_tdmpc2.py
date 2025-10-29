"""
Recording-enabled TD-MPC2 agent for mechanistic interpretability.

This module provides RecordingTDMPC2, a subclass of TDMPC2 that can record:
1. Episode data (observations, actions, rewards, latent states)
2. Planning data (MPPI iterations, elite trajectories, action distributions)

Both are optional - you can record episodes only, planning only, or both.
"""

import torch
from tdmpc2 import TDMPC2


class RecordingTDMPC2(TDMPC2):
    """
    TD-MPC2 agent that records episode and/or planning data.
    
    Unified recording interface for mechanistic interpretability:
    - episode_recorder: Records obs, actions, rewards, latent states to disk
    - planning_recorder: Records MPPI planning details (iterations, elites, etc.)
    
    Both recorders are optional. Use ActivationPatcher separately if you need
    to modify activations in real-time.
    
    Example:
        from common.episode_data_recorder import EpisodeDataRecorder
        from common.planning_data_recorder import PlanningDataRecorder
        from recording_tdmpc2 import RecordingTDMPC2
        
        # Setup recorders
        episode_rec = EpisodeDataRecorder(cfg)
        planning_rec = PlanningDataRecorder()
        
        # Create recording agent
        agent = RecordingTDMPC2(cfg, 
                               episode_recorder=episode_rec,
                               planning_recorder=planning_rec)
        
        # Run episodes - both are recorded automatically
        obs = env.reset()
        action = agent.act(obs, t0=True, eval_mode=True)
    """

    def __init__(self, cfg, episode_recorder=None, planning_recorder=None):
        """
        Initialize recording TD-MPC2 agent.
        
        Args:
            cfg: Configuration object
            episode_recorder: Optional EpisodeDataRecorder instance
            planning_recorder: Optional PlanningDataRecorder instance
        """
        super().__init__(cfg)
        self.episode_recorder = episode_recorder
        self.planning_recorder = planning_recorder

        # Track last latent for episode recording
        self._last_latent = None

        print(f"RecordingTDMPC2 initialized:")
        print(
            f"  Episode recording: {'enabled' if episode_recorder else 'disabled'}"
        )
        print(
            f"  Planning recording: {'enabled' if planning_recorder else 'disabled'}"
        )

    @torch.no_grad()
    def act(self, obs, t0=False, eval_mode=False, task=None):
        """
        Select action and record episode data.
        
        This overrides the parent act() to capture latent states for episode recording.
        Planning recording happens inside _plan_with_recording().
        
        Args:
            obs: Observation from environment
            t0: Whether this is first timestep
            eval_mode: Whether in evaluation mode
            task: Task index (for multi-task)
            
        Returns:
            action: Action to take
        """
        # Prepare observation
        obs_tensor = obs.to(self.device, non_blocking=True).unsqueeze(0)
        if task is not None:
            task_tensor = torch.tensor([task], device=self.device)
        else:
            task_tensor = None

        # Encode observation to latent state
        # (ActivationPatcher hooks will run here if attached)
        latent_state = self.model.encode(obs_tensor, task_tensor)

        # Store latent for episode recording
        self._last_latent = latent_state.detach()

        # Get action via planning or policy
        if self.cfg.mpc:
            # Use planning with recording
            action = self._plan_with_recording(obs_tensor, latent_state, t0,
                                               eval_mode, task_tensor)
        else:
            # Use policy directly (no planning to record)
            action, info = self.model.pi(latent_state, task_tensor)
            if eval_mode:
                action = info["mean"]

        return action.cpu()

    def get_last_latent(self):
        """Return the latent state from the last act() call."""
        return self._last_latent

    @torch.no_grad()
    def _plan_with_recording(self, obs, latent_state, t0, eval_mode, task):
        """
        Plan using MPPI and optionally record planning details.
        
        If planning_recorder is None, this just calls the parent _plan().
        Otherwise, it records mean/std evolution, elite trajectories, etc.
        """
        if self.planning_recorder is None:
            # No recording, use parent's _plan
            return self.plan(obs, t0=t0, eval_mode=eval_mode, task=task)

        # Planning with recording - replicate _plan() logic with recording
        z = latent_state

        # Sample policy trajectories (if using policy prior)
        if self.cfg.num_pi_trajs > 0:
            pi_actions = torch.empty(self.cfg.horizon,
                                     self.cfg.num_pi_trajs,
                                     self.cfg.action_dim,
                                     device=self.device)
            _z = z.repeat(self.cfg.num_pi_trajs, 1)
            for t in range(self.cfg.horizon - 1):
                pi_actions[t], _ = self.model.pi(_z, task)
                _z = self.model.next(_z, pi_actions[t], task)
            pi_actions[-1], _ = self.model.pi(_z, task)

        # Initialize state and parameters
        z = z.repeat(self.cfg.num_samples, 1)
        mean = torch.zeros(self.cfg.horizon,
                           self.cfg.action_dim,
                           device=self.device)
        std = torch.full((self.cfg.horizon, self.cfg.action_dim),
                         self.cfg.max_std,
                         dtype=torch.float,
                         device=self.device)
        if not t0:
            mean[:-1] = self._prev_mean[1:]
        actions = torch.empty(self.cfg.horizon,
                              self.cfg.num_samples,
                              self.cfg.action_dim,
                              device=self.device)
        if self.cfg.num_pi_trajs > 0:
            actions[:, :self.cfg.num_pi_trajs] = pi_actions

        # Store iterations data for recording
        iterations_data = []

        # Iterate MPPI
        for iteration in range(self.cfg.iterations):
            # Sample actions
            r = torch.randn(self.cfg.horizon,
                            self.cfg.num_samples - self.cfg.num_pi_trajs,
                            self.cfg.action_dim,
                            device=std.device)
            actions_sample = mean.unsqueeze(1) + std.unsqueeze(1) * r
            actions_sample = actions_sample.clamp(-1, 1)
            actions[:, self.cfg.num_pi_trajs:] = actions_sample
            if self.cfg.multitask:
                actions = actions * self.model._action_masks[task]

            # Compute elite actions
            value = self._estimate_value(z, actions, task).nan_to_num(0)
            elite_idxs = torch.topk(value.squeeze(1),
                                    self.cfg.num_elites,
                                    dim=0).indices
            elite_value, elite_actions = value[elite_idxs], actions[:,
                                                                    elite_idxs]

            # Record this iteration (before updating mean/std)
            iterations_data.append({
                'mean_actions': mean.clone(),
                'std_actions': std.clone(),
                'elite_actions': elite_actions.permute(
                    1, 0, 2).clone(),  # [num_elites, horizon, action_dim]
                'elite_values': elite_value.squeeze(-1).clone(),
            })

            # Update parameters
            max_value = elite_value.max(0).values
            score = torch.exp(self.cfg.temperature * (elite_value - max_value))
            score = score / score.sum(0)
            mean = (score.unsqueeze(0) *
                    elite_actions).sum(dim=1) / (score.sum(0) + 1e-9)
            std = ((score.unsqueeze(0) *
                    (elite_actions - mean.unsqueeze(1))**2).sum(dim=1) /
                   (score.sum(0) + 1e-9)).sqrt()
            std = std.clamp(self.cfg.min_std, self.cfg.max_std)
            if self.cfg.multitask:
                mean = mean * self.model._action_masks[task]
                std = std * self.model._action_masks[task]

        # Select final action
        rand_idx = torch.distributions.Categorical(score.squeeze(1)).sample()
        actions_final = elite_actions[:, rand_idx]
        a, std_final = actions_final[0], std[0]
        if not eval_mode:
            a = a + std_final * torch.randn(self.cfg.action_dim,
                                            device=std_final.device)
        self._prev_mean.copy_(mean)
        final_action = a.clamp(-1, 1)

        # Record planning step
        self.planning_recorder.record_planning_step(
            initial_latent=latent_state,
            iterations_data=iterations_data,
            selected_action=final_action,
            all_sampled_actions=actions
            if self.planning_recorder.save_full_trajectories else None)

        return final_action
