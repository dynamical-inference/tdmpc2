"""
Projected TD-MPC2 agent for low-rank latent space experiments.

Applies a low-rank projection Pk(z) = μ + Uk @ Uk.T @ (z - μ)
after encoding observations. Planning proceeds normally from the projected latent.
"""

import torch
from tdmpc2 import TDMPC2


class ProjectedTDMPC2(TDMPC2):
    """
    TD-MPC2 agent with low-rank latent projection after encoding.
    
    Applies projection Pk(z) after encoding:
        z̃₀ = Pk(encoder(o₀))
    
    Planning then proceeds normally from z̃₀.
    
    Example:
        from common.latent_projector import LatentProjector
        from projected_tdmpc2 import ProjectedTDMPC2
        agent = ProjectedTDMPC2(cfg, projector=projector)
        agent.load(checkpoint_path)
        # Use normally - projection applied after encoding
        action = agent.act(obs, t0=True, eval_mode=True)
    """

    def __init__(self, cfg, projector):
        """
        Initialize projected TD-MPC2 agent.
        
        Args:
            cfg: Configuration object
            projector: LatentProjector instance (must be provided)
        """
        super().__init__(cfg)
        if projector is None:
            raise ValueError("projector must be provided.")
        self.projector = projector

        # For compatibility with run_episode_with_recording (sweeps/utils.py)
        # Stores dict with 'raw', 'projected', 'pca_components'
        self._last_latent = None
        self.planning_recorder = None  # Not used, but needed for compatibility

    def get_last_latent(self):
        """
        Return latent states from the last act() call.
        
        Returns:
            dict with keys:
                - 'raw': Raw latent from encoder, shape (1, latent_dim)
                - 'projected': Projected latent, shape (1, latent_dim)
                - 'pca_components': PCA coordinates, shape (1, k)
        """
        if self._last_latent is None:
            raise RuntimeError(
                "get_last_latent() called before act(). "
                "Call act() first to encode and project an observation.")
        return self._last_latent

    def set_projector(self, projector):
        """Set or update the projector."""
        if projector is None:
            raise ValueError("projector must be provided.")
        self.projector = projector

    def _project(self, z):
        """Always apply projection using the provided projector."""
        return self.projector(z)

    @torch.no_grad()
    def act(self, obs, t0=False, eval_mode=False, task=None):
        """
        Select an action with projection applied after encoding.
        
        Args:
            obs: Observation from the environment
            t0: Whether this is the first observation in the episode
            eval_mode: Whether to use the mean of the action distribution
            task: Task index (only used for multi-task experiments)
            
        Returns:
            Action to take in the environment
        """
        obs = obs.to(self.device, non_blocking=True).unsqueeze(0)
        if task is not None:
            task = torch.tensor([task], device=self.device)

        # Encode and project
        z_raw = self.model.encode(obs, task)
        z_proj, pca_components = self.projector.project_and_components(z_raw)

        # Store all three for get_last_latent()
        self._last_latent = {
            'raw': z_raw.detach(),
            'projected': z_proj.detach(),
            'pca_components': pca_components.detach(),
        }

        z = z_proj  # Use projected latent for planning

        if self.cfg.mpc:
            # Plan from projected latent
            return self._plan_from_latent(z,
                                          t0=t0,
                                          eval_mode=eval_mode,
                                          task=task).cpu()

        # Non-MPC mode: use policy directly
        action, info = self.model.pi(z, task)
        if eval_mode:
            action = info["mean"]
        return action[0].cpu()

    @torch.no_grad()
    def encode_and_project(self, obs, task=None):
        """
        Encode observation and apply projection. Useful for analysis.
        
        Args:
            obs: Observation tensor
            task: Task index
            
        Returns:
            dict with keys:
                - 'raw': Raw latent from encoder
                - 'projected': Projected latent
                - 'pca_components': PCA coordinates (k-dimensional)
        """
        obs = obs.to(self.device)
        if obs.ndim == 1:
            obs = obs.unsqueeze(0)
        if task is not None:
            task = torch.tensor([task], device=self.device)

        z_raw = self.model.encode(obs, task)
        z_proj, pca_components = self.projector.project_and_components(z_raw)

        return {
            'raw': z_raw,
            'projected': z_proj,
            'pca_components': pca_components,
        }
