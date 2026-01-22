"""
Nonlinear low-rank projection for latent space using MLP autoencoder.

Finds the minimum dimensionality needed to reconstruct latents by training:
    z → Encoder → h (k-dim) → Decoder → z_reconstructed

The projection is: z_proj = decoder(encoder(z))
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional, List
from tqdm import tqdm


class MLPEncoder(nn.Module):
    """MLP encoder: z (latent_dim) → h (k)"""

    def __init__(self,
                 latent_dim: int,
                 k: int,
                 hidden_dims: List[int] = [256, 128]):
        super().__init__()
        layers = []
        in_dim = latent_dim
        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(in_dim, h_dim),
                nn.LayerNorm(h_dim),
                nn.GELU(),
            ])
            in_dim = h_dim
        layers.append(nn.Linear(in_dim, k))
        self.net = nn.Sequential(*layers)

    def forward(self, z):
        return self.net(z)


class MLPDecoder(nn.Module):
    """MLP decoder: h (k) → z_reconstructed (latent_dim)"""

    def __init__(self,
                 k: int,
                 latent_dim: int,
                 hidden_dims: List[int] = [128, 256]):
        super().__init__()
        layers = []
        in_dim = k
        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(in_dim, h_dim),
                nn.LayerNorm(h_dim),
                nn.GELU(),
            ])
            in_dim = h_dim
        layers.append(nn.Linear(in_dim, latent_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, h):
        return self.net(h)


class NonlinearLatentProjector:
    """
    Nonlinear low-rank projection using MLP autoencoder.
    
    Learns a nonlinear mapping:
        z → encoder → h (k-dim bottleneck) → decoder → z_proj
    
    Useful for finding the intrinsic dimensionality of the latent manifold.
    
    Example:
        # Fit autoencoder
        projector = NonlinearLatentProjector.fit(
            latents, k=8, epochs=1000, lr=1e-3
        )
        
        # Project
        z_proj = projector(z)
        h = projector.get_components(z)  # k-dim representation
        
        # Check reconstruction quality
        mse = projector.reconstruction_mse(latents)
    """

    def __init__(self,
                 encoder: nn.Module,
                 decoder: nn.Module,
                 mean: torch.Tensor,
                 std: torch.Tensor,
                 device: str = 'cuda:0'):
        """
        Initialize with trained encoder/decoder.
        
        Args:
            encoder: Trained encoder network
            decoder: Trained decoder network
            mean: Mean of training latents (for normalization)
            std: Std of training latents (for normalization)
            device: Device to run on
        """
        self.device = torch.device(device)
        self.encoder = encoder.to(self.device)
        self.decoder = decoder.to(self.device)
        self.mean = mean.to(self.device)
        self.std = std.to(self.device)
        self.k = encoder.net[-1].out_features
        self.latent_dim = decoder.net[-1].out_features

        self.encoder.eval()
        self.decoder.eval()

    def __call__(self, z):
        """Apply nonlinear projection."""
        return self.project(z)

    @torch.no_grad()
    def project(self, z):
        """
        Project z through autoencoder bottleneck.
        
        z_proj = decoder(encoder(z))
        """
        z = z.to(self.device)
        # Normalize
        z_norm = (z - self.mean) / self.std
        # Encode → Decode
        h = self.encoder(z_norm)
        z_recon_norm = self.decoder(h)
        # Denormalize
        z_proj = z_recon_norm * self.std + self.mean
        return z_proj

    @torch.no_grad()
    def get_components(self, z):
        """
        Get the k-dimensional bottleneck representation.
        
        Returns shape (..., k)
        """
        z = z.to(self.device)
        z_norm = (z - self.mean) / self.std
        return self.encoder(z_norm)

    @torch.no_grad()
    def project_and_components(self, z):
        """
        Project z and return both projected latent and k-dim components.
        """
        z = z.to(self.device)
        z_norm = (z - self.mean) / self.std
        h = self.encoder(z_norm)
        z_recon_norm = self.decoder(h)
        z_proj = z_recon_norm * self.std + self.mean
        return z_proj, h

    @classmethod
    def fit(cls,
            train_latents,
            test_latents,
            k: int,
            hidden_dims: List[int] = [256, 128],
            epochs: int = 1000,
            batch_size: int = 256,
            lr: float = 1e-3,
            weight_decay: float = 1e-5,
            early_stopping_patience: Optional[int] = None,
            device: str = 'cuda:0',
            verbose: bool = True):
        """
        Fit autoencoder to find k-dimensional nonlinear subspace.
        
        Args:
            train_latents: Training latents, shape (N_train, latent_dim)
            test_latents: Test latents, shape (N_test, latent_dim)
            k: Bottleneck dimension
            hidden_dims: Hidden layer sizes for encoder (decoder mirrors)
            epochs: Number of training epochs
            batch_size: Batch size for training
            lr: Learning rate
            weight_decay: Weight decay for optimizer
            early_stopping_patience: Stop if test loss doesn't improve for N epochs.
                                     None disables early stopping.
            device: Device to train on
            verbose: Print progress
            
        Returns:
            NonlinearLatentProjector instance
        """
        device = torch.device(device)

        if isinstance(train_latents, np.ndarray):
            train_latents = torch.from_numpy(train_latents).float()
        if isinstance(test_latents, np.ndarray):
            test_latents = torch.from_numpy(test_latents).float()

        train_latents = train_latents.to(device)
        test_latents = test_latents.to(device)

        N_train = train_latents.shape[0]
        N_test = test_latents.shape[0]
        latent_dim = train_latents.shape[1]

        # Normalize using TRAINING stats only
        mean = train_latents.mean(dim=0)
        std = train_latents.std(dim=0) + 1e-8
        train_norm = (train_latents - mean) / std
        test_norm = (test_latents - mean) / std

        # Create encoder/decoder
        encoder = MLPEncoder(latent_dim, k, hidden_dims).to(device)
        decoder = MLPDecoder(k, latent_dim,
                             list(reversed(hidden_dims))).to(device)

        # Optimizer
        params = list(encoder.parameters()) + list(decoder.parameters())
        optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, epochs)

        # Training loop
        train_losses = []
        test_losses = []
        best_test_loss = float('inf')
        best_state = None
        patience_counter = 0

        iterator = tqdm(range(epochs),
                        desc=f"Fitting k={k}") if verbose else range(epochs)

        for epoch in iterator:
            # Training
            encoder.train()
            decoder.train()

            perm = torch.randperm(N_train, device=device)
            epoch_loss = 0.0
            n_batches = 0

            for i in range(0, N_train, batch_size):
                idx = perm[i:i + batch_size]
                z_batch = train_norm[idx]

                h = encoder(z_batch)
                z_recon = decoder(h)
                loss = ((z_batch - z_recon)**2).mean()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                n_batches += 1

            scheduler.step()
            train_loss = epoch_loss / n_batches
            train_losses.append(train_loss)

            # Evaluate on test set
            encoder.eval()
            decoder.eval()
            with torch.no_grad():
                h_test = encoder(test_norm)
                z_recon_test = decoder(h_test)
                test_loss = ((test_norm - z_recon_test)**2).mean().item()
            test_losses.append(test_loss)

            # Early stopping check
            if test_loss < best_test_loss:
                best_test_loss = test_loss
                best_state = {
                    'encoder': {
                        k: v.cpu().clone()
                        for k, v in encoder.state_dict().items()
                    },
                    'decoder': {
                        k: v.cpu().clone()
                        for k, v in decoder.state_dict().items()
                    },
                }
                patience_counter = 0
            else:
                patience_counter += 1

            if early_stopping_patience and patience_counter >= early_stopping_patience:
                if verbose:
                    print(f"\nEarly stopping at epoch {epoch + 1}")
                break

            if verbose and (epoch + 1) % 100 == 0:
                iterator.set_postfix(train=f"{train_loss:.6f}",
                                     test=f"{test_loss:.6f}")

        # Load best model if early stopping was used
        if best_state is not None and early_stopping_patience:
            encoder.load_state_dict({
                k: v.to(device) for k, v in best_state['encoder'].items()
            })
            decoder.load_state_dict({
                k: v.to(device) for k, v in best_state['decoder'].items()
            })

        encoder.eval()
        decoder.eval()

        # Compute final metrics
        with torch.no_grad():
            # Train R²
            h_train = encoder(train_norm)
            z_recon_train = decoder(h_train)
            train_var = train_norm.var(dim=0).sum().item()
            train_res_var = ((train_norm -
                              z_recon_train)**2).mean(dim=0).sum().item()
            train_r2 = 1.0 - train_res_var / train_var

            # Test R²
            h_test = encoder(test_norm)
            z_recon_test = decoder(h_test)
            test_var = test_norm.var(dim=0).sum().item()
            test_res_var = ((test_norm -
                             z_recon_test)**2).mean(dim=0).sum().item()
            test_r2 = 1.0 - test_res_var / test_var

        print(f"NonlinearLatentProjector.fit: k={k}, "
              f"train_R²={train_r2:.4f}, test_R²={test_r2:.4f} "
              f"(N_train={N_train}, N_test={N_test})")

        projector = cls(encoder, decoder, mean, std, device=str(device))
        projector._train_losses = train_losses
        projector._test_losses = test_losses
        projector._train_r2 = train_r2
        projector._test_r2 = test_r2
        return projector

    @torch.no_grad()
    def reconstruction_mse(self, latents):
        """Compute MSE on given latents."""
        if isinstance(latents, np.ndarray):
            latents = torch.from_numpy(latents).float()
        latents = latents.to(self.device)

        z_proj = self.project(latents)
        mse = ((latents - z_proj)**2).mean().item()
        return mse

    @torch.no_grad()
    def reconstruction_r2(self, latents):
        """Compute R² (explained variance ratio) on given latents."""
        if isinstance(latents, np.ndarray):
            latents = torch.from_numpy(latents).float()
        latents = latents.to(self.device)

        z_proj = self.project(latents)
        total_var = ((latents - latents.mean(dim=0))**2).sum()
        residual_var = ((latents - z_proj)**2).sum()
        r2 = 1.0 - residual_var / total_var
        return r2.item()

    def save(self, path):
        """Save projector to disk."""
        torch.save(
            {
                'encoder_state': self.encoder.state_dict(),
                'decoder_state': self.decoder.state_dict(),
                'mean': self.mean.cpu(),
                'std': self.std.cpu(),
                'k': self.k,
                'latent_dim': self.latent_dim,
                'encoder_config': {
                    'latent_dim': self.latent_dim,
                    'k': self.k,
                    'hidden_dims': self._get_hidden_dims(self.encoder),
                },
            }, path)
        print(f"NonlinearLatentProjector saved to {path}")

    def _get_hidden_dims(self, encoder):
        """Extract hidden dims from encoder."""
        dims = []
        for m in encoder.net:
            if isinstance(m, nn.Linear) and m.out_features != self.k:
                dims.append(m.out_features)
        return dims

    @classmethod
    def load(cls, path, device='cuda:0'):
        """Load projector from disk."""
        data = torch.load(path, map_location='cpu', weights_only=False)

        config = data['encoder_config']
        encoder = MLPEncoder(config['latent_dim'], config['k'],
                             config['hidden_dims'])
        decoder = MLPDecoder(config['k'], config['latent_dim'],
                             list(reversed(config['hidden_dims'])))

        encoder.load_state_dict(data['encoder_state'])
        decoder.load_state_dict(data['decoder_state'])

        projector = cls(encoder,
                        decoder,
                        data['mean'],
                        data['std'],
                        device=device)
        print(f"NonlinearLatentProjector loaded: k={projector.k}, "
              f"latent_dim={projector.latent_dim}")
        return projector

    def __repr__(self):
        return (f"NonlinearLatentProjector(k={self.k}, "
                f"latent_dim={self.latent_dim}, device={self.device})")
