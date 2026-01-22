"""
Low-rank projection for latent space.

Implements Pk(z) = μ + Uk @ Uk.T @ (z - μ)
where μ is the mean and Uk are orthonormal directions found via:
  - PCA (unsupervised): directions of maximum variance
  - CCA (supervised): directions maximally correlated with state
  - RRR (supervised): ridge regression directions predictive of state
  - hybrid_cca/hybrid_rrr: k_supervised supervised directions + (k - k_supervised)
    PCA directions from the residual space orthogonal to supervised subspace
"""

import torch
import numpy as np
from typing import Literal, Optional


class LatentProjector:
    """
    Low-rank projection for latent space.
    
    Projects latent vectors onto a k-dimensional subspace:
        Pk(z) = μ + Uk @ Uk.T @ (z - μ)
    
    Supports multiple fitting methods:
        - 'pca': Unsupervised, directions of maximum variance
        - 'cca': Supervised, directions maximally correlated with state
        - 'rrr': Supervised, ridge regression directions predictive of state
        - 'hybrid_cca': k_supervised CCA directions + (k - k_supervised) residual PCA
        - 'hybrid_rrr': k_supervised RRR directions + (k - k_supervised) residual PCA
    
    Example (unsupervised):
        projector = LatentProjector.fit(latents, k=32, method='pca')
        
    Example (supervised):
        projector = LatentProjector.fit(
            latents, k=5, method='cca', states=states, ridge=1e-4
        )
        
    Example (hybrid):
        projector = LatentProjector.fit(
            latents, k=32, method='hybrid_cca',
            states=states, k_supervised=5, ridge=1e-4
        )
    """

    def __init__(self,
                 mean,
                 components,
                 method='pca',
                 state_mean=None,
                 state_std=None,
                 device='cuda:0'):
        """
        Initialize projector with pre-computed mean and components.
        
        Args:
            mean: Mean of training latents, shape (latent_dim,)
            components: Orthonormal directions, shape (latent_dim, k)
            method: Fitting method used ('pca', 'cca', 'rrr')
            state_mean: Mean of states (for supervised methods)
            state_std: Std of states (for supervised methods)
            device: Device to store tensors on
        """
        self.device = torch.device(device)
        self.mean = mean.to(self.device)
        self.Uk = components.to(self.device)
        self.k = self.Uk.shape[1]
        self.latent_dim = self.Uk.shape[0]
        self.method = method

        # For supervised methods
        if state_mean is not None:
            self.state_mean = state_mean.to(self.device)
            self.state_std = state_std.to(self.device)
        else:
            self.state_mean = None
            self.state_std = None

    def __call__(self, z):
        """Apply projection. Alias for project()."""
        return self.project(z)

    def project(self, z, U=None):
        """
        Project z onto the low-rank subspace.
        
        Pk(z) = μ + Uk @ Uk.T @ (z - μ)
        """

        if U is None:
            U = self.Uk
        z_centered = z - self.mean
        z_proj = (z_centered @ U) @ U.T
        return self.mean + z_proj

    def get_components(self, z):
        """
        Get the k-dimensional coordinates of z in the subspace.
        
        Returns shape (..., k)
        """
        z_centered = z - self.mean
        return z_centered @ self.Uk

    def project_and_components(self, z):
        """
        Project z and return both projected latent and k-dim components.
        
        Returns:
            Tuple of (z_projected, components)
        """
        z_centered = z - self.mean
        components = z_centered @ self.Uk
        z_proj = components @ self.Uk.T
        return self.mean + z_proj, components

    # =========================================================================
    # Fitting methods
    # =========================================================================

    @classmethod
    def fit(cls,
            latents,
            k: int,
            method: Literal['pca', 'cca', 'rrr', 'hybrid_cca',
                            'hybrid_rrr'] = 'pca',
            states: Optional[np.ndarray] = None,
            k_supervised: Optional[int] = None,
            ridge: float = 1e-4,
            device: str = 'cuda:0'):
        """
        Fit projection from latents (and optionally states).
        
        Args:
            latents: Training latents, shape (N, latent_dim)
            k: Total number of components
            method: Fitting method:
                - 'pca': unsupervised PCA
                - 'cca': supervised CCA (max k = state_dim)
                - 'rrr': supervised ridge reduced rank regression (max k = state_dim)
                - 'hybrid_cca': k_supervised CCA + (k - k_supervised) residual PCA
                - 'hybrid_rrr': k_supervised RRR + (k - k_supervised) residual PCA
            states: Training states, shape (N, state_dim). Required for supervised.
            k_supervised: Number of supervised directions (for hybrid methods).
                          Defaults to state_dim if not specified.
            ridge: Ridge regularization (for supervised methods)
            device: Device for the projector
            
        Returns:
            LatentProjector instance
        """
        if isinstance(latents, np.ndarray):
            latents = torch.from_numpy(latents).float()

        z_mean = latents.mean(dim=0)
        Z = latents - z_mean

        if method == 'pca':
            Uk, info = cls._fit_pca(Z, k)
            return cls(z_mean, Uk, method='pca', device=device)

        elif method in ('cca', 'rrr'):
            if states is None:
                raise ValueError(f"method='{method}' requires states argument")
            if isinstance(states, np.ndarray):
                states = torch.from_numpy(states).float()

            s_mean = states.mean(dim=0)
            s_std = states.std(dim=0) + 1e-8
            S = (states - s_mean) / s_std

            if method == 'cca':
                Uk, info = cls._fit_cca(Z, S, k, ridge)
            else:
                Uk, info = cls._fit_rrr(Z, S, k, ridge)

            return cls(z_mean,
                       Uk,
                       method=method,
                       state_mean=s_mean,
                       state_std=s_std,
                       device=device)

        elif method in ('hybrid_cca', 'hybrid_rrr'):
            if states is None:
                raise ValueError(f"method='{method}' requires states argument")
            if isinstance(states, np.ndarray):
                states = torch.from_numpy(states).float()

            s_mean = states.mean(dim=0)
            s_std = states.std(dim=0) + 1e-8
            S = (states - s_mean) / s_std

            # Default k_supervised to state_dim
            state_dim = S.shape[1]
            if k_supervised is None:
                k_supervised = state_dim
            k_supervised = min(k_supervised, state_dim, k)

            # Get supervised method type
            supervised_method = 'cca' if method == 'hybrid_cca' else 'rrr'
            Uk, info = cls._fit_hybrid(Z, S, k, k_supervised, supervised_method,
                                       ridge)

            return cls(z_mean,
                       Uk,
                       method=method,
                       state_mean=s_mean,
                       state_std=s_std,
                       device=device)

        else:
            raise ValueError(f"Unknown method: {method}")

    @staticmethod
    def _fit_pca(Z, k):
        """
        PCA: find directions of maximum variance.
        
        Args:
            Z: Centered latents (N, latent_dim)
            k: Number of components
            
        Returns:
            Uk: shape (latent_dim, k)
            info: dict with fitting info
        """
        U, S, Vh = torch.linalg.svd(Z, full_matrices=False)
        Uk = Vh[:k].T

        total_var = (S**2).sum()
        explained_var = (S[:k]**2).sum() / total_var
        print(
            f"LatentProjector.fit (pca): k={k}, explained_var={explained_var:.4f}"
        )

        return Uk, {
            'explained_var': explained_var.item(),
            'singular_values': S[:k]
        }

    @staticmethod
    def _fit_cca(Z, S, k, ridge):
        """
        CCA: find directions maximally correlated with states.
        
        Solves for directions u in Z-space such that corr(Z @ u, S @ v) is maximized.
        
        Args:
            Z: Centered latents (N, latent_dim)
            S: Standardized states (N, state_dim)
            k: Number of components (capped at state_dim)
            ridge: Ridge regularization
            
        Returns:
            Uk: shape (latent_dim, k)
            info: dict with fitting info
        """
        N = Z.shape[0]
        k = min(k, S.shape[1])

        # Covariance matrices
        Czz = (Z.T @ Z) / N + ridge * torch.eye(Z.shape[1])
        Css = (S.T @ S) / N + ridge * torch.eye(S.shape[1])
        Czs = (Z.T @ S) / N

        # Whiten Z and S
        Lzz = torch.linalg.cholesky(Czz)
        Lzz_inv = torch.linalg.inv(Lzz)
        Lss = torch.linalg.cholesky(Css)
        Lss_inv = torch.linalg.inv(Lss)

        # Whitened cross-covariance
        M = Lzz_inv @ Czs @ Lss_inv.T

        # SVD gives canonical directions
        U, canonical_corr, Vh = torch.linalg.svd(M, full_matrices=False)

        # Transform back to original Z-space
        Uk = Lzz_inv.T @ U[:, :k]
        Uk, _ = torch.linalg.qr(Uk)  # Orthonormalize

        print(f"LatentProjector.fit (cca): k={k}, "
              f"canonical_corr={canonical_corr[:k].tolist()}")

        return Uk, {'canonical_correlations': canonical_corr[:k]}

    @staticmethod
    def _fit_rrr(Z, S, k, ridge):
        """
        Ridge Reduced Rank Regression: ridge regression + orthonormal basis.
        
        1. B = (Z^T Z + λI)^{-1} Z^T S
        2. Uk = orthonormal basis for column space of B (via SVD)
        
        Args:
            Z: Centered latents (N, latent_dim)
            S: Standardized states (N, state_dim)
            k: Number of components (capped at state_dim)
            ridge: Ridge regularization
            
        Returns:
            Uk: shape (latent_dim, k)
            info: dict with fitting info
        """
        latent_dim = Z.shape[1]
        k = min(k, S.shape[1])

        # Ridge regression
        ZtZ = Z.T @ Z + ridge * torch.eye(latent_dim)
        ZtS = Z.T @ S
        B = torch.linalg.solve(ZtZ, ZtS)  # (latent_dim, state_dim)

        # Extract orthonormal basis for column space
        U, singular_values, Vh = torch.linalg.svd(B, full_matrices=False)
        Uk = U[:, :k]
        Uk, _ = torch.linalg.qr(Uk)

        print(f"LatentProjector.fit (rrr): k={k}, "
              f"singular_values={singular_values[:k].tolist()}")

        return Uk, {'singular_values': singular_values[:k]}

    @classmethod
    def _fit_hybrid(cls, Z, S, k, k_supervised, supervised_method, ridge):
        """
        Hybrid: supervised directions + residual PCA completion.
        
        1. Find k_supervised supervised directions (CCA or RRR)
        2. Project Z onto the orthogonal complement of the supervised subspace
        3. Find (k - k_supervised) PCA directions in the residual space
        4. Concatenate to form the full k-dimensional basis
        
        Args:
            Z: Centered latents (N, latent_dim)
            S: Standardized states (N, state_dim)
            k: Total number of components
            k_supervised: Number of supervised directions
            supervised_method: 'cca' or 'rrr'
            ridge: Ridge regularization
            
        Returns:
            Uk: shape (latent_dim, k)
            info: dict with fitting info
        """
        # Step 1: Get supervised directions
        if supervised_method == 'cca':
            U_sup, sup_info = cls._fit_cca(Z, S, k_supervised, ridge)
        else:
            U_sup, sup_info = cls._fit_rrr(Z, S, k_supervised, ridge)

        k_residual = k - k_supervised
        if k_residual <= 0:
            # No residual PCA needed
            print(f"LatentProjector.fit (hybrid_{supervised_method}): "
                  f"k_supervised={k_supervised}, k_residual=0")
            return U_sup, sup_info

        # Step 2: Project Z onto orthogonal complement of supervised subspace
        # Z_residual = Z - Z @ U_sup @ U_sup.T
        Z_proj_sup = Z @ U_sup @ U_sup.T
        Z_residual = Z - Z_proj_sup

        # Step 3: PCA on residual
        U_res, S_res, Vh_res = torch.linalg.svd(Z_residual, full_matrices=False)
        U_pca_residual = Vh_res[:k_residual].T  # (latent_dim, k_residual)

        # Step 4: Ensure orthogonality to supervised directions
        # Gram-Schmidt orthogonalization against U_sup
        U_pca_orth = U_pca_residual - U_sup @ (U_sup.T @ U_pca_residual)
        U_pca_orth, _ = torch.linalg.qr(U_pca_orth)

        # Step 5: Concatenate
        Uk = torch.cat([U_sup, U_pca_orth], dim=1)

        # Compute residual explained variance
        total_var = (S_res**2).sum()
        residual_explained = (S_res[:k_residual]**
                              2).sum() / total_var if total_var > 0 else 0

        print(f"LatentProjector.fit (hybrid_{supervised_method}): "
              f"k_supervised={k_supervised}, k_residual={k_residual}, "
              f"residual_explained_var={residual_explained:.4f}")

        return Uk, {
            'supervised_info': sup_info,
            'k_supervised': k_supervised,
            'k_residual': k_residual,
            'residual_explained_var': residual_explained
        }

    # =========================================================================
    # Evaluation methods
    # =========================================================================

    def explained_variance_ratio(self, latents):
        """Fraction of latent variance explained by the k-dimensional subspace."""
        if isinstance(latents, np.ndarray):
            latents = torch.from_numpy(latents).float().to(self.device)

        centered = latents - self.mean
        total_var = (centered**2).sum()

        projected = self.project(latents)
        residual = latents - projected
        residual_var = (residual**2).sum()

        return (1.0 - residual_var / total_var).item()

    def state_prediction_r2(self, latents, states):
        """
        R² for predicting states from the projected subspace (supervised methods).
        
        Fits a linear decoder from k-dim components to states and returns R².
        """
        if isinstance(latents, np.ndarray):
            latents = torch.from_numpy(latents).float().to(self.device)
        if isinstance(states, np.ndarray):
            states = torch.from_numpy(states).float().to(self.device)

        components = self.get_components(latents)

        # Standardize states if we have normalization
        if self.state_mean is not None:
            states_norm = (states - self.state_mean) / self.state_std
        else:
            states_norm = states - states.mean(dim=0)

        # Fit linear decoder: W = (C^T C)^{-1} C^T S
        CtC = components.T @ components + 1e-6 * torch.eye(self.k,
                                                           device=self.device)
        CtS = components.T @ states_norm
        W = torch.linalg.solve(CtC, CtS)  # (k, state_dim)

        # Predict and compute R²
        pred = components @ W
        ss_res = ((states_norm - pred)**2).sum()
        ss_tot = ((states_norm - states_norm.mean(dim=0))**2).sum()

        return (1.0 - ss_res / ss_tot).item()

    # =========================================================================
    # Persistence
    # =========================================================================

    def save(self, path):
        """Save projector to disk."""
        data = {
            'mean': self.mean.cpu(),
            'Uk': self.Uk.cpu(),
            'k': self.k,
            'latent_dim': self.latent_dim,
            'method': self.method,
        }
        if self.state_mean is not None:
            data['state_mean'] = self.state_mean.cpu()
            data['state_std'] = self.state_std.cpu()
        torch.save(data, path)
        print(f"LatentProjector ({self.method}) saved to {path}")

    @classmethod
    def load(cls, path, device='cuda:0'):
        """Load projector from disk."""
        data = torch.load(path, map_location='cpu', weights_only=True)
        projector = cls(mean=data['mean'],
                        components=data['Uk'],
                        method=data.get('method', 'pca'),
                        state_mean=data.get('state_mean'),
                        state_std=data.get('state_std'),
                        device=device)
        print(f"LatentProjector loaded: method={projector.method}, "
              f"k={projector.k}, latent_dim={projector.latent_dim}")
        return projector

    def __repr__(self):
        return (f"LatentProjector(method={self.method}, k={self.k}, "
                f"latent_dim={self.latent_dim}, device={self.device})")
