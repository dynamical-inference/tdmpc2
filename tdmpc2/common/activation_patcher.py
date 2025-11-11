"""
Activation patcher for mechanistic interpretability of TD-MPC2 agents.
Enables causal interventions like activation patching, dimension ablation, etc.

Uses PyTorch forward hooks to intercept and modify activations in real-time.
Compatible with EpisodeDataRecorder: patched activations will automatically
be captured by the episode recorder if both are used together.

Example usage:
    # Create activation patcher
    patcher = ActivationPatcher(agent.model)
    
    # Ablate dimension 42 in encoder output
    patcher.add_ablation('encoder_output', dims=[42])
    
    # Run episode with intervention
    obs = env.reset()
    action = agent.act(obs)  # Dimension 42 will be zeroed automatically
    
    # Clear interventions
    patcher.clear_interventions()
"""

import torch
import torch.nn as nn
from typing import Dict, List, Callable, Optional, Union
import numpy as np
from collections import defaultdict


class ActivationPatcher:
    """
    Patches (modifies) activations in real-time for causal interventions.
    
    Uses PyTorch forward hooks to intercept activations during the forward pass
    and apply modifications like dimension ablation, noise injection, etc.
    
    Compatible with EpisodeDataRecorder: if both are used together, the recorder
    will automatically capture the MODIFIED activations after patches are applied.
    """

    def __init__(
        self,
        model,
        modules_to_hook: List[str] = [
            'encoder_output',  # Latent state z
            'dynamics_output',  # Predicted next latent
            'reward_output',  # Predicted reward
            'pi_output',  # Policy output
            'q_output',  # Q-function output
        ]):
        """
        Initialize activation patcher.

        Args:
            model: WorldModel instance to attach hooks to
        """
        self.model = model
        self.hooks = []
        self.activations = {}  # Stores recorded activations
        self.interventions = {}  # Stores intervention functions
        self.enabled = True  # Global enable/disable
        self.modules_to_hook = modules_to_hook

        # Setup hooks on all key components
        self._setup_hooks()

        print("ActivationPatcher initialized with hooks on:")
        for module in self.modules_to_hook:
            print(f"  - {module}")

    def _setup_hooks(self):
        """Register forward hooks on world model components."""

        for module in self.modules_to_hook:
            print(f"Setting up hooks for {module}")
            if module == 'encoder_output':
                #NOTE(R): _encoder is a ModuleDict (for multitask experiments), so we need to register hooks on individual
                # encoders (e.g., 'state' or 'rgb').
                for key, encoder in self.model._encoder.items():
                    self.hooks.append(
                        encoder.register_forward_hook(
                            self._make_hook('encoder_output')))
            else:
                self.hooks.append(
                    getattr(self.model, module).register_forward_hook(
                        self._make_hook(module)))

        # if hasattr(self.model, '_encoder'):
        #     # Register hooks on each encoder in the ModuleDict (e.g., 'state' or 'rgb')
        #     for key, encoder in self.model._encoder.items():
        #         self.hooks.append(
        #             encoder.register_forward_hook(
        #                 self._make_hook('encoder_output')))

        # # Hook dynamics model
        # if hasattr(self.model, '_dynamics'):
        #     self.hooks.append(
        #         self.model._dynamics.register_forward_hook(
        #             self._make_hook('dynamics_output')))

        # # Hook reward model
        # if hasattr(self.model, '_reward'):
        #     self.hooks.append(
        #         self.model._reward.register_forward_hook(
        #             self._make_hook('reward_output')))

        # Hook policy
        # if hasattr(self.model, '_pi'):
        #     self.hooks.append(
        #         self.model._pi.register_forward_hook(
        #             self._make_hook('pi_output')))

        # Hook Q-functions (just record, hard to intervene on ensemble)
        # if hasattr(self.model, '_Qs'):
        #     self.hooks.append(
        #         self.model._Qs.register_forward_hook(
        #             self._make_hook('q_output')))

    def _make_hook(self, name: str):
        """
        Create a forward hook that records and optionally modifies activations.

        Args:
            name: Name identifier for this hook location

        Returns:
            Hook function
        """

        def hook(module, input, output):
            if not self.enabled:
                return output

            # Record the original activation
            self.activations[name] = output.detach().clone()

            # Apply intervention if specified
            if name in self.interventions:
                intervention_fn = self.interventions[name]
                # Apply intervention and return modified output
                output = intervention_fn(output)

            return output

        return hook

    def add_intervention(self, location: str, intervention_fn: Callable):
        """
        Add a custom intervention function at a specific location.

        Args:
            location: Where to intervene ('encoder_output', 'dynamics_output', etc.)
            intervention_fn: Function that takes activation tensor and returns modified tensor
                           Example: lambda x: x * 0.5
        """
        self.interventions[location] = intervention_fn
        print(f"Added intervention at: {location}")

    def add_ablation(self, location: str, dims: Union[int, List[int]]):
        """
        Ablate (zero out) specific dimensions at a location.

        Args:
            location: Where to ablate ('encoder_output', 'dynamics_output', etc.)
            dims: Dimension index or list of indices to zero out

        Example:
            recorder.add_ablation('encoder_output', dims=[0, 5, 42])
        """
        if isinstance(dims, int):
            dims = [dims]

        def ablation_fn(activation):
            # Clone to avoid modifying original
            modified = activation.clone()
            # Zero out specified dimensions
            if activation.dim() == 2:  # [batch, features]
                modified[:, dims] = 0
            elif activation.dim() == 3:  # [seq, batch, features]
                modified[:, :, dims] = 0
            return modified

        self.add_intervention(location, ablation_fn)
        print(f"  Ablating dimensions: {dims}")

    def add_replacement(self, location: str, replacement_value: torch.Tensor,
                        dims: List[int]):
        """
        Replace activations with a specific value.

        Args:
            location: Where to replace
            replacement_value: Tensor to replace with (must be broadcastable to activation sub-tensor, shape typically [..., len(dims)])
            dims: List of dimensions to replace

        Example:
            # Replace with mean activation from baseline
            mean_activation = baseline_activations['encoder_output'].mean(0)
            # mean_activation shape: [features]
            recorder.add_replacement('encoder_output', mean_activation, dims=[0,1,2])
        """

        # replacement_value should be a 1D tensor of features, e.g. shape [features], or a tensor that can be indexed over dims.
        def replacement_fn(activation):
            modified = activation.clone()
            # Replace only specific dimensions
            if activation.dim() == 2:
                # activation: [batch, features], replacement_value: [features] (or broadcastable)
                modified[:, dims] = replacement_value[dims]
            elif activation.dim() == 3:
                modified[:, :, dims] = replacement_value[dims]
            return modified

        self.add_intervention(location, replacement_fn)
        print(
            f"  Replacing with fixed value (dims: {dims}), replacement_value shape: {tuple(replacement_value.shape)}"
        )

    def add_noise(self, location: str, dims: List[int], scale: float = 0.1):
        """
        Add Gaussian noise to activations.

        Args:
            location: Where to add noise
            scale: Standard deviation of noise
            dims: List of dimensions to add noise to
        """

        def noise_fn(activation):
            modified = activation.clone()
            noise = torch.randn_like(activation) * scale

            # Only add noise to specific dimensions
            mask = torch.zeros_like(activation)
            if activation.dim() == 2:
                mask[:, dims] = 1
            elif activation.dim() == 3:
                mask[:, :, dims] = 1
            noise = noise * mask

            return modified + noise

        self.add_intervention(location, noise_fn)
        print(f"  Adding noise (scale={scale}) (dims: {dims})")

    # def add_scaling(self,
    #                 location: str,
    #                 scale: float,
    #                 dims: Optional[List[int]] = None):
    #     """
    #     Scale activations by a factor.

    #     Args:
    #         location: Where to scale
    #         scale: Multiplicative factor
    #         dims: Optional list of dimensions to scale (None = all dims)
    #     """

    #     def scaling_fn(activation):
    #         modified = activation.clone()

    #         if dims is not None:
    #             if activation.dim() == 2:
    #                 modified[:, dims] = modified[:, dims] * scale
    #             elif activation.dim() == 3:
    #                 modified[:, :, dims] = modified[:, :, dims] * scale
    #         else:
    #             modified = modified * scale

    #         return modified

    #     self.add_intervention(location, scaling_fn)
    #     dims_str = f" (dims: {dims})" if dims else ""
    #     print(f"  Scaling by {scale}{dims_str}")

    def clear_interventions(self, location: Optional[str] = None):
        """
        Clear interventions.

        Args:
            location: Specific location to clear (None = clear all)
        """
        if location is None:
            self.interventions = {}
            print("Cleared all interventions")
        else:
            if location in self.interventions:
                del self.interventions[location]
                print(f"Cleared intervention at: {location}")

    def enable(self):
        """Enable all interventions."""
        self.enabled = True
        print("Interventions enabled")

    def disable(self):
        """Disable all interventions (but keep them defined)."""
        self.enabled = False
        print("Interventions disabled")

    def get_activation(self, location: str) -> Optional[torch.Tensor]:
        """
        Get recorded activation at a specific location.

        Args:
            location: Location name

        Returns:
            Activation tensor or None if not recorded yet
        """
        return self.activations.get(location, None)

    def get_all_activations(self) -> Dict[str, torch.Tensor]:
        """Get all recorded activations."""
        return self.activations.copy()

    def remove_hooks(self):
        """Remove all hooks from the model."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
        print("Removed all hooks")

    def __del__(self):
        """Cleanup hooks when object is destroyed."""
        self.remove_hooks()

    def summary(self):
        """Print summary of current interventions."""
        print("\n" + "=" * 60)
        print("ActivationPatcher Summary")
        print("=" * 60)
        print(f"Enabled: {self.enabled}")
        print(f"Number of interventions: {len(self.interventions)}")
        if self.interventions:
            print("\nActive interventions:")
            for location in self.interventions:
                print(f"  - {location}")
        else:
            print("\nNo active interventions")
        print("=" * 60 + "\n")


# class DimensionImportanceExperiment:
#     """
#     Run systematic dimension ablation experiments to identify important dimensions.
#     """

#     def __init__(self, env, agent, activation_patcher: ActivationPatcher):
#         """
#         Args:
#             env: Environment instance
#             agent: TD-MPC2 agent
#             activation_patcher: ActivationPatcher instance
#         """
#         self.env = env
#         self.agent = agent
#         self.patcher = activation_patcher

#     def test_dimension_importance(self,
#                                   location: str = 'encoder_output',
#                                   n_dims: Optional[int] = None,
#                                   dims_to_test: Optional[List[int]] = None,
#                                   num_episodes: int = 1) -> Dict[int, float]:
#         """
#         Test importance of each dimension by ablation.

#         Args:
#             location: Where to ablate ('encoder_output', etc.)
#             n_dims: Number of dimensions to test (None = all)
#             dims_to_test: Specific dimensions to test (overrides n_dims)
#             num_episodes: Number of episodes per test

#         Returns:
#             Dictionary mapping dimension index to importance score (performance drop)
#         """
#         print("\n" + "=" * 60)
#         print(f"Testing Dimension Importance at: {location}")
#         print("=" * 60)

#         # Get baseline performance (no intervention)
#         print("\nRunning baseline (no intervention)...")
#         self.patcher.disable()
#         baseline_rewards = []
#         for _ in range(num_episodes):
#             reward = self._run_episode()
#             baseline_rewards.append(reward)
#         baseline_reward = np.mean(baseline_rewards)
#         print(
#             f"Baseline reward: {baseline_reward:.2f} ± {np.std(baseline_rewards):.2f}"
#         )

#         # Determine which dimensions to test
#         if dims_to_test is None:
#             if n_dims is None:
#                 # Need to infer from first forward pass
#                 print("\nInferring number of dimensions...")
#                 self.patcher.enable()
#                 self._run_episode(max_steps=10)
#                 activation = self.patcher.get_activation(location)
#                 if activation is None:
#                     raise ValueError(f"No activation recorded at {location}")
#                 n_dims = activation.shape[-1]
#                 print(f"Detected {n_dims} dimensions")

#             dims_to_test = list(range(n_dims))

#         # Test each dimension
#         importance_scores = {}
#         self.patcher.enable()

#         print(f"\nTesting {len(dims_to_test)} dimensions...")
#         for i, dim in enumerate(dims_to_test):
#             # Set up ablation for this dimension
#             self.patcher.clear_interventions()
#             self.patcher.add_ablation(location, dims=[dim])

#             # Run episodes
#             rewards = []
#             for _ in range(num_episodes):
#                 reward = self._run_episode()
#                 rewards.append(reward)

#             mean_reward = np.mean(rewards)
#             importance = baseline_reward - mean_reward
#             importance_scores[dim] = importance

#             if (i + 1) % 10 == 0 or (i + 1) == len(dims_to_test):
#                 print(f"  Tested {i+1}/{len(dims_to_test)} dimensions")

#         # Clear interventions
#         self.patcher.clear_interventions()

#         # Print results
#         print("\n" + "=" * 60)
#         print("Results Summary")
#         print("=" * 60)
#         sorted_dims = sorted(importance_scores.items(),
#                              key=lambda x: abs(x[1]),
#                              reverse=True)
#         print("\nTop 10 most important dimensions:")
#         for dim, importance in sorted_dims[:10]:
#             direction = "↓" if importance > 0 else "↑"
#             print(
#                 f"  Dim {dim:3d}: {direction} {abs(importance):6.2f} reward change"
#             )
#         print("=" * 60 + "\n")

#         return importance_scores

#     def _run_episode(self, max_steps: Optional[int] = None) -> float:
#         """
#         Run a single episode and return total reward.

#         Args:
#             max_steps: Maximum steps (None = until done)

#         Returns:
#             Total episode reward
#         """
#         obs = self.env.reset()
#         done = False
#         total_reward = 0
#         step = 0

#         while not done:
#             action = self.agent.act(obs, t0=(step == 0), eval_mode=True)
#             obs, reward, done, info = self.env.step(action)
#             total_reward += reward
#             step += 1

#             if max_steps is not None and step >= max_steps:
#                 break

#         return total_reward

# def compare_interventions(env, agent, activation_patcher: ActivationPatcher,
#                           interventions: List[tuple]) -> Dict[str, float]:
#     """
#     Compare multiple interventions.

#     Args:
#         env: Environment
#         agent: Agent
#         activation_patcher: ActivationPatcher instance
#         interventions: List of (name, setup_fn) tuples where setup_fn configures intervention

#     Returns:
#         Dictionary mapping intervention name to reward

#     Example:
#         interventions = [
#             ('baseline', lambda p: p.disable()),
#             ('ablate_dim_42', lambda p: p.add_ablation('encoder_output', [42])),
#             ('ablate_dim_100', lambda p: p.add_ablation('encoder_output', [100])),
#         ]
#         results = compare_interventions(env, agent, patcher, interventions)
#     """
#     results = {}

#     for name, setup_fn in interventions:
#         # Clear previous interventions
#         activation_patcher.clear_interventions()

#         # Setup this intervention
#         setup_fn(activation_patcher)

#         # Run episode
#         obs = env.reset()
#         done = False
#         total_reward = 0
#         t = 0

#         while not done:
#             action = agent.act(obs, t0=(t == 0), eval_mode=True)
#             obs, reward, done, info = env.step(action)
#             total_reward += reward
#             t += 1

#         results[name] = total_reward
#         print(f"{name}: {total_reward:.2f}")

#     return results
