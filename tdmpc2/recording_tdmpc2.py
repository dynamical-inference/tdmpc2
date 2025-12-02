"""
Recording-enabled TD-MPC2 agent for mechanistic interpretability.

This module provides RecordingTDMPC2, a subclass of TDMPC2 that can record:
1. Episode data (observations, actions, rewards, latent states)
2. Planning data (MPPI iterations, elite trajectories, action distributions)

Both are optional - you can record episodes only, planning only, or both.
"""

import torch
from tdmpc2 import TDMPC2
from common import math


class RecordingTDMPC2(TDMPC2):
    """
    TD-MPC2 agent that records episode and/or planning data.
    
    Unified recording interface for mechanistic interpretability:
    - episode_recorder: Records obs, actions, rewards, latent states to disk
    - planning_recorder: Records MPPI planning details (iterations, elites, etc.)
    - record_rollouts: Optionally records dynamics model rollouts (latents & rewards)
    
    All recording options are optional. Use ActivationPatcher separately if you need
    to modify activations in real-time.
    
    Example:
        from common.episode_data_recorder import EpisodeDataRecorder
        from common.planning_data_recorder import PlanningDataRecorder
        from recording_tdmpc2 import RecordingTDMPC2
        
        # Setup recorders
        episode_rec = EpisodeDataRecorder(cfg)
        planning_rec = PlanningDataRecorder(
            save_dir='recordings',
            record_rollouts=True,
            record_only_last_iteration=False  # Set True to save only last iteration
        )
        
        # Create recording agent
        agent = RecordingTDMPC2(cfg, planning_recorder=planning_rec)
        
        # Run episodes - all data is recorded automatically
        obs = env.reset()
        action = agent.act(obs, t0=True, eval_mode=True)
    """

    def __init__(
        self,
        cfg,
        planning_recorder=None,
    ):
        """
        Initialize recording TD-MPC2 agent.
        
        Args:
            cfg: Configuration object
            planning_recorder: Optional PlanningDataRecorder instance
        """
        super().__init__(cfg)
        self.planning_recorder = planning_recorder

        # Track last latent for episode recording
        self._last_latent = None

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
    def _estimate_value(self, z, actions, task, return_rollouts=False):
        """
        Estimate value of trajectories and optionally capture rollouts.
        
        This overrides the parent's _estimate_value to optionally record
        the dynamics model rollouts (latent states and rewards) during
        value estimation, avoiding duplicate computation.
        
        Args:
            z: Initial latent state (num_samples, latent_dim)
            actions: Action sequences (horizon, num_samples, action_dim)
            task: Task tensor
            return_rollouts: If True, also return rollout data
            
        Returns:
            If return_rollouts=False: Just the value estimates (num_samples, 1)
            If return_rollouts=True: Tuple of (values, rollout_data) where rollout_data
                contains 'latents' (horizon+1, num_samples, latent_dim) and 
                'rewards' (horizon, num_samples, 1)
        """
        from common import math

        G, discount = 0, 1
        termination = torch.zeros(self.cfg.num_samples,
                                  1,
                                  dtype=torch.float32,
                                  device=z.device)

        # Optionally store rollout data
        if return_rollouts:
            latents = torch.zeros(self.cfg.horizon + 1,
                                  self.cfg.num_samples,
                                  z.shape[-1],
                                  device=z.device)
            rewards = torch.zeros(self.cfg.horizon,
                                  self.cfg.num_samples,
                                  1,
                                  device=z.device)
            latents[0] = z

        # Roll out dynamics
        for t in range(self.cfg.horizon):
            reward_raw = self.model.reward(
                z, actions[t],
                task)  #reward_raw.shape = (num_samples=512, num_bins=101)
            reward = math.two_hot_inv(
                reward_raw, self.cfg)  #reward.shape = (num_samples=512, 1)
            z = self.model.next(
                z, actions[t],
                task)  #z.shape = (num_samples=512, latent_dim=512)

            if return_rollouts:
                rewards[t] = reward
                latents[t + 1] = z

            G = G + discount * (
                1 - termination) * reward  #G.shape = (num_samples=512, 1)
            discount_update = self.discount[torch.tensor(
                task)] if self.cfg.multitask else self.discount
            discount = discount * discount_update  #discount.shape = (1,)

            if self.cfg.episodic:
                termination = torch.clip(
                    termination +
                    (self.model.termination(z, task) > 0.5).float(),
                    max=1.)

        # Final Q-value

        # NOTE (R): Remember that z is getting replaced with the next latent ( z = self.model.next(z, actions[t], task))
        #  so the value is calculated with the last z (zt+h).

        # NOTE (R): The action to evaluate as "final return" is computed from the policy prior, not the elite actions,
        # using the last z (zt+h).

        action, _ = self.model.pi(
            z, task)  #action.shape = (num_samples=512, action_dim=1)
        value = G + discount * (1 - termination) * self.model.Q(
            z, action, task,
            return_type='avg')  #value.shape = (num_samples=512, 1)
        #self.model.Q(z, action, task,return_type='avg').shape = (num_samples=512, 1)

        if return_rollouts:
            rollout_data = {'latents': latents, 'rewards': rewards}
            return value, rollout_data
        else:
            return value

    @torch.no_grad()
    def _plan_with_recording(self, obs, z, t0, eval_mode, task):
        """
        Plan using MPPI and optionally record planning details.
        
        If planning_recorder is None, this just calls the parent plan().
        Otherwise, it records mean/std evolution, elite trajectories, etc.
        
        If record_rollouts is True, also records dynamics model rollouts for
        elite trajectories (predicted latent states and rewards at each timestep).
        """
        if self.planning_recorder is None:
            # No recording, use parent's plan
            return self.plan(obs, t0=t0, eval_mode=eval_mode, task=task)

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

        # NOTE (R): pi_actions are the actions sampled from the policy prior
        # pi_actions.shape = (horizon=3, num_pi_trajs=24, action_dim=1)

        # Initialize state and parameters
        z = z.repeat(self.cfg.num_samples,
                     1)  #z.shape = (num_samples=512, latent_dim=512)
        mean = torch.zeros(
            self.cfg.horizon, self.cfg.action_dim,
            device=self.device)  #mean.shape = (horizon=3, action_dim=1)
        std = torch.full(
            (self.cfg.horizon, self.cfg.action_dim),
            self.cfg.max_std,
            dtype=torch.float,
            device=self.device)  #std.shape = (horizon=3, action_dim=1)
        if not t0:
            mean[:-1] = self._prev_mean[1:]
        actions = torch.empty(
            self.cfg.horizon,
            self.cfg.num_samples,
            self.cfg.action_dim,
            device=self.device
        )  #actions.shape = (horizon=3, num_samples=512, action_dim=1)
        if self.cfg.num_pi_trajs > 0:
            actions[:, :self.cfg.num_pi_trajs] = pi_actions

        # (R): Store iterations data for recording
        iterations_data = []

        # Iterate MPPI
        for _ in range(self.cfg.iterations):
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

            if self.planning_recorder.record_rollouts:
                value, rollout_data_all = self._estimate_value(
                    z, actions, task, return_rollouts=True)
                value = value.nan_to_num(0)
            else:
                value = self._estimate_value(z, actions, task).nan_to_num(0)
                rollout_data_all = None

            #NOTE (R):
            #value.shape = (num_samples=512, 1)
            #rollout_data_all['latents'].shape = (horizon+1=4, num_samples=512, latent_dim=512)
            #rollout_data_all['rewards'].shape = (horizon=3, num_samples=512, num_bins=101)
            #mean.shape = (horizon=3, action_dim=1)

            elite_idxs = torch.topk(
                value.squeeze(1), self.cfg.num_elites,
                dim=0).indices  #elite_idxs.shape = (num_elites=64,) integers
            elite_value, elite_actions = value[elite_idxs], actions[:,
                                                                    elite_idxs]
            #elite_value.shape = (num_elites=64, 1)
            #elite_actions.shape = (horizon=3, num_elites=64, action_dim=1)

            # Update parameters
            max_value = elite_value.max(0).values  #max_value.shape = (1,)
            score = torch.exp(self.cfg.temperature * (elite_value - max_value))
            score = score / score.sum(0)  #score.shape = (num_elites=64, 1)
            mean = (score.unsqueeze(0) * elite_actions).sum(dim=1) / (
                score.sum(0) + 1e-9)  #mean.shape = (horizon=3, action_dim=1)
            std = ((score.unsqueeze(0) *
                    (elite_actions - mean.unsqueeze(1))**2).sum(dim=1) /
                   (score.sum(0) +
                    1e-9)).sqrt()  #std.shape = (horizon=3, action_dim=1)
            std = std.clamp(self.cfg.min_std, self.cfg.max_std)
            if self.cfg.multitask:
                mean = mean * self.model._action_masks[task]
                std = std * self.model._action_masks[task]

            # Record planning
            iteration_record = {
                'mean_actions': mean.clone(),
                'std_actions': std.clone(),
                'elite_actions': elite_actions,
                'elite_values': elite_value,
                'score': score.clone(),
            }
            if rollout_data_all is not None:
                iteration_record['rollout_elite_latents'] = rollout_data_all[
                    'latents'][:, elite_idxs]
                iteration_record['rollout_elite_rewards'] = rollout_data_all[
                    'rewards'][:, elite_idxs]
            iterations_data.append(iteration_record)

        # Select action
        rand_idx = math.gumbel_softmax_sample(
            score.squeeze(1))  # e.g. tensor(58, device='cuda:0')
        actions = torch.index_select(elite_actions, 1, rand_idx).squeeze(
            1)  # shape (horizon=3, action_dim=1)
        a, std = actions[0], std[0]
        if not eval_mode:
            a = a + std * torch.randn(self.cfg.action_dim, device=std.device)
        self._prev_mean.copy_(mean)
        final_action = a.clamp(-1, 1)

        # Record planning step
        self.planning_recorder.record_planning_step(
            iterations_data=iterations_data,
            final_actions=actions,
            final_action=final_action,
            random_idx=rand_idx,
        )

        return final_action
