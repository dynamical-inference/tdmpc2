"""
Activation patcher for mechanistic interpretability of TD-MPC2 agents.
Enables causal interventions like activation patching, dimension ablation, 
directional ablation, and more.

Uses PyTorch forward hooks to intercept and modify activations in real-time.
Compatible with EpisodeDataRecorder: patched activations will automatically
be captured by the episode recorder if both are used together.

Example usage - Dimension ablation:
    # Create activation patcher
    patcher = ActivationPatcher(agent.model)
    
    # Ablate dimension 42 in encoder output
    patcher.add_ablation('encoder_output', dims=[42])
    
    # Run episode with intervention
    obs = env.reset()
    action = agent.act(obs)  # Dimension 42 will be zeroed automatically
    
    # Clear interventions
    patcher.clear_interventions()

Example usage - Directional ablation:
    # Extract a meaningful direction (e.g., from PCA or learned features)
    velocity_direction = torch.randn(512)  # latent_dim = 512
    
    # Store the direction for reuse
    patcher.store_direction('velocity', velocity_direction)
    
    # Remove the velocity component from latent states
    patcher.add_directional_ablation('encoder_output', direction_name='velocity')
    
    # Or scale it down instead of removing completely
    patcher.add_directional_scaling('encoder_output', scale=0.5, direction_name='velocity')

Example usage - Directional addition (for steering/control):
    # Extract an amplitude direction from episodes (e.g., pole angle oscillation)
    amplitude_direction = extract_amplitude_direction(episodes)  # Your extraction method
    
    # Store the direction
    patcher.store_direction('amplitude', amplitude_direction)
    
    # Add the direction to INCREASE oscillation amplitude
    patcher.add_directional_addition('encoder_output', magnitude=2.0, direction_name='amplitude')
    
    # Or DECREASE oscillation amplitude
    patcher.add_directional_addition('encoder_output', magnitude=-1.0, direction_name='amplitude')
    
    # Can also control multiple features simultaneously
    patcher.add_directional_addition('encoder_output',
                                      magnitude=0.0,  # unused
                                      direction_names=['amplitude', 'frequency'],
                                      magnitudes=[2.0, -0.5])

Advanced directional ablation examples:
    # Multi-directional ablation (remove multiple components simultaneously)
    position_dir = extract_feature_direction('position')  # Your extraction method
    velocity_dir = extract_feature_direction('velocity')
    patcher.add_directional_ablation('encoder_output', 
                                       directions=[position_dir, velocity_dir])
    
    # Clamp directional component within bounds (e.g., prevent extreme values)
    patcher.add_directional_clamp('encoder_output',
                                    min_magnitude=-2.0,
                                    max_magnitude=2.0,
                                    direction_name='velocity')
    
    # Reverse a directional component (scale by -1)
    patcher.add_directional_scaling('encoder_output', scale=-1.0, direction_name='force')
    
    # Direct intervention without storing (useful for one-off experiments)
    temp_direction = compute_gradient_based_direction()  # Your method
    patcher.add_directional_ablation('encoder_output', direction=temp_direction)
    
    # Check current state
    patcher.summary()  # Shows all interventions and stored directions

Example usage - Step-gated interventions:
    # Apply ablation only during specific steps of an episode
    patcher.add_directional_ablation('encoder_output',
                                       direction_name='velocity',
                                       start_step=10,  # Start at step 10
                                       end_step=50)    # Stop after step 50
    
    # Run episode with step-gated intervention
    obs = env.reset()
    patcher.reset_step()  # Reset step counter at episode start
    for step in range(max_steps):
        patcher.set_step(step)  # Update step counter
        action = agent.act(obs)  # Intervention only applies during steps 10-50
        obs, reward, done, info = env.step(action)
        if done:
            break
"""

import torch
from typing import Dict, List, Callable, Optional, Union, Tuple
import warnings


class ActivationPatcher:
    """
    Patches (modifies) activations in real-time for causal interventions.
    
    Uses PyTorch forward hooks to intercept activations during the forward pass
    and apply modifications like:
    - Dimension ablation: Zero out specific coordinate dimensions
    - Directional ablation: Remove components along meaningful direction vectors
    - Directional scaling: Amplify or attenuate directional components
    - Directional addition: Add/inject components to steer behavior (key for control)
    - Directional replacement: Set directional components to fixed values
    - Directional clamping: Constrain directional magnitudes to bounds
    - Noise injection: Add Gaussian noise to dimensions
    - Custom interventions: Apply arbitrary functions to activations
    
    Directional interventions operate on the vector space of activations, projecting
    onto learned or extracted directions (e.g., from PCA, sparse autoencoders, or 
    interpretable features) for more semantically meaningful causal analysis.
    
    Compatible with EpisodeDataRecorder: if both are used together, the recorder
    will automatically capture the MODIFIED activations after patches are applied.

    Layer-level hooks:
        Pass hook_encoder_layers=True and/or hook_dynamics_layers=True to record
        outputs at every layer. Layer activations are stored under keys like:
            encoder_layer/<obs_key>/<layer_idx>
            dynamics_layer/<layer_idx>
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
            ],
            hook_encoder_layers: bool = False,
            hook_dynamics_layers: bool = False):
        """
        Initialize activation patcher.

        Args:
            model: WorldModel instance to attach hooks to
            hook_encoder_layers: If True, record every encoder layer output
            hook_dynamics_layers: If True, record every dynamics layer output
        """
        self.model = model
        self.hooks = []
        self.activations = {}  # Stores recorded activations
        self.interventions = {}  # Stores intervention functions
        self.enabled = True  # Global enable/disable
        self.modules_to_hook = modules_to_hook
        self.hook_encoder_layers = hook_encoder_layers
        self.hook_dynamics_layers = hook_dynamics_layers
        self.directions = {
        }  # Stores named direction vectors for directional interventions
        self.current_step = 0  # Current step counter for step-gated interventions

        # Setup hooks on all key components
        self._setup_hooks()

        print("ActivationPatcher initialized with hooks on:")
        for module in self.modules_to_hook:
            print(f"  - {module}")
        if self.hook_encoder_layers:
            print("  - encoder_layers/*")
        if self.hook_dynamics_layers:
            print("  - dynamics_layers/*")

    def _setup_hooks(self):
        """Register forward hooks on world model components."""
        module_map = {
            'dynamics_output': '_dynamics',
            'reward_output': '_reward',
            'pi_output': '_pi',
            'q_output': '_Qs',
        }

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
                attr_name = module_map.get(module, module)
                if not hasattr(self.model, attr_name):
                    warnings.warn(
                        f"Model has no attribute '{attr_name}' for hook '{module}'. Skipping."
                    )
                    continue
                self.hooks.append(
                    getattr(self.model, attr_name).register_forward_hook(
                        self._make_hook(module)))

        if self.hook_encoder_layers:
            self._setup_encoder_layer_hooks()
        if self.hook_dynamics_layers:
            self._setup_dynamics_layer_hooks()

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

    def _setup_encoder_layer_hooks(self):
        """Register hooks on every encoder layer."""
        if not hasattr(self.model, '_encoder'):
            warnings.warn("Model has no _encoder; cannot hook encoder layers.")
            return

        if "state" not in self.model._encoder:
            warnings.warn(
                "Encoder layer hooks only support 'state' observations.")
            return

        for key, encoder in self.model._encoder.items():
            if key != "state":
                continue
            prefix = f"encoder_layer/{key}"
            self._register_layer_hooks(encoder, prefix)

    def _setup_dynamics_layer_hooks(self):
        """Register hooks on every dynamics layer."""
        if not hasattr(self.model, '_dynamics'):
            warnings.warn(
                "Model has no _dynamics; cannot hook dynamics layers.")
            return
        self._register_layer_hooks(self.model._dynamics, "dynamics_layer")

    def _register_layer_hooks(self, module, prefix: str):
        """
        Register hooks on immediate children of a module.

        For Sequential encoders/MLPs, this captures each layer output.
        """
        for name, layer in module.named_children():
            hook_name = f"{prefix}/{name}"
            self.hooks.append(
                layer.register_forward_hook(self._make_hook(hook_name)))

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

    def add_ablation(self,
                     location: str,
                     dims: Union[int, List[int]],
                     start_step: Optional[int] = None,
                     end_step: Optional[int] = None):
        """
        Ablate (zero out) specific dimensions at a location.

        Args:
            location: Where to ablate ('encoder_output', 'dynamics_output', etc.)
            dims: Dimension index or list of indices to zero out
            start_step: Step at which to start applying the intervention (inclusive, None = from beginning)
            end_step: Step at which to stop applying the intervention (inclusive, None = until end)

        Example:
            recorder.add_ablation('encoder_output', dims=[0, 5, 42])
            
            # Only ablate during steps 10-50
            recorder.add_ablation('encoder_output', dims=[0, 5], start_step=10, end_step=50)
        """
        if isinstance(dims, int):
            dims = [dims]

        def ablation_fn(activation):
            # Check step bounds
            if start_step is not None and self.current_step < start_step:
                return activation
            if end_step is not None and self.current_step > end_step:
                return activation

            # Clone to avoid modifying original
            modified = activation.clone()
            # Zero out specified dimensions
            if activation.dim() == 2:  # [batch, features]
                modified[:, dims] = 0
            elif activation.dim() == 3:  # [seq, batch, features]
                modified[:, :, dims] = 0
            return modified

        self.add_intervention(location, ablation_fn)
        step_str = self._format_step_bounds(start_step, end_step)
        print(f"  Ablating dimensions: {dims}{step_str}")

    def add_replacement(self,
                        location: str,
                        replacement_value: torch.Tensor,
                        dims: List[int],
                        start_step: Optional[int] = None,
                        end_step: Optional[int] = None):
        """
        Replace activations with a specific value.

        Args:
            location: Where to replace
            replacement_value: Tensor to replace with (must be broadcastable to activation sub-tensor, shape typically [..., len(dims)])
            dims: List of dimensions to replace
            start_step: Step at which to start applying the intervention (inclusive, None = from beginning)
            end_step: Step at which to stop applying the intervention (inclusive, None = until end)

        Example:
            # Replace with mean activation from baseline
            mean_activation = baseline_activations['encoder_output'].mean(0)
            # mean_activation shape: [features]
            recorder.add_replacement('encoder_output', mean_activation, dims=[0,1,2])
        """

        # replacement_value should be a 1D tensor of features, e.g. shape [features], or a tensor that can be indexed over dims.
        def replacement_fn(activation):
            # Check step bounds
            if start_step is not None and self.current_step < start_step:
                return activation
            if end_step is not None and self.current_step > end_step:
                return activation

            modified = activation.clone()
            # Replace only specific dimensions
            if activation.dim() == 2:
                # activation: [batch, features], replacement_value: [features] (or broadcastable)
                modified[:, dims] = replacement_value[dims]
            elif activation.dim() == 3:
                modified[:, :, dims] = replacement_value[dims]
            return modified

        self.add_intervention(location, replacement_fn)
        step_str = self._format_step_bounds(start_step, end_step)
        print(
            f"  Replacing with fixed value (dims: {dims}), replacement_value shape: {tuple(replacement_value.shape)}{step_str}"
        )

    def add_noise(self,
                  location: str,
                  dims: List[int],
                  scale: float = 0.1,
                  start_step: Optional[int] = None,
                  end_step: Optional[int] = None):
        """
        Add Gaussian noise to activations.

        Args:
            location: Where to add noise
            scale: Standard deviation of noise
            dims: List of dimensions to add noise to
            start_step: Step at which to start applying the intervention (inclusive, None = from beginning)
            end_step: Step at which to stop applying the intervention (inclusive, None = until end)
        """

        def noise_fn(activation):
            # Check step bounds
            if start_step is not None and self.current_step < start_step:
                return activation
            if end_step is not None and self.current_step > end_step:
                return activation

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
        step_str = self._format_step_bounds(start_step, end_step)
        print(f"  Adding noise (scale={scale}) (dims: {dims}){step_str}")

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

    # ========================================================================
    # DIRECTIONAL INTERVENTION METHODS
    # ========================================================================

    def _normalize_direction(self, direction: torch.Tensor) -> torch.Tensor:
        """
        Normalize a direction vector to unit length.
        
        Args:
            direction: Direction vector of shape [features]
            
        Returns:
            Normalized direction vector
        """
        norm = torch.norm(direction)
        if norm < 1e-8:
            warnings.warn(
                "Direction vector has near-zero norm. Returning as-is.")
            return direction
        return direction / norm

    def _project_onto_direction(self, activation: torch.Tensor,
                                direction: torch.Tensor) -> torch.Tensor:
        """
        Project activation onto a direction vector.
        
        Computes the component of activation along direction:
            proj_d(x) = (x · d) * d
        
        Args:
            activation: Activation tensor of shape [..., features]
            direction: Direction vector of shape [features] (assumed normalized)
            
        Returns:
            Projection tensor, same shape as activation
        """
        # Compute scalar projection: (x · d)
        # activation: [..., features], direction: [features]
        scalar_proj = torch.sum(activation * direction, dim=-1, keepdim=True)

        # Multiply by direction to get vector projection
        # scalar_proj: [..., 1], direction: [features] -> result: [..., features]
        vector_proj = scalar_proj * direction

        return vector_proj

    def _remove_direction_component(self, activation: torch.Tensor,
                                    direction: torch.Tensor) -> torch.Tensor:
        """
        Remove the component of activation along a direction (orthogonal projection).
        
        Computes: x - proj_d(x) = x - (x · d) * d
        
        Args:
            activation: Activation tensor of shape [..., features]
            direction: Direction vector of shape [features] (assumed normalized)
            
        Returns:
            Activation with direction component removed
        """
        projection = self._project_onto_direction(activation, direction)
        return activation - projection

    def store_direction(self,
                        name: str,
                        direction: torch.Tensor,
                        normalize: bool = True):
        """
        Store a named direction vector for later use in interventions.
        
        Args:
            name: Identifier for this direction (e.g., 'velocity', 'pca_component_0')
            direction: Direction vector of shape [features]
            normalize: Whether to normalize to unit length (recommended)
            
        Example:
            >>> velocity_dir = torch.randn(512)  # latent_dim = 512
            >>> patcher.store_direction('velocity', velocity_dir)
            >>> patcher.add_directional_ablation('encoder_output', direction_name='velocity')
        """
        if normalize:
            direction = self._normalize_direction(direction)

        self.directions[name] = direction.detach().clone()
        print(f"Stored direction '{name}' with shape {tuple(direction.shape)}")

    def add_directional_ablation(
            self,
            location: str,
            direction: Optional[torch.Tensor] = None,
            direction_name: Optional[str] = None,
            directions: Optional[List[torch.Tensor]] = None,
            direction_names: Optional[List[str]] = None,
            normalize: bool = True,
            start_step: Optional[int] = None,
            end_step: Optional[int] = None):
        """
        Ablate (remove) components along one or more direction vectors.
        
        This performs orthogonal projection: x_new = x - (x · d) * d
        which removes the component of x along direction d.
        
        Args:
            location: Where to ablate ('encoder_output', 'dynamics_output', etc.)
            direction: Single direction vector of shape [features] (mutually exclusive with direction_name)
            direction_name: Name of stored direction to use (mutually exclusive with direction)
            directions: List of direction vectors (for multi-directional ablation)
            direction_names: List of stored direction names (for multi-directional ablation)
            normalize: Whether to normalize direction(s) to unit length
            start_step: Step at which to start applying the intervention (inclusive, None = from beginning)
            end_step: Step at which to stop applying the intervention (inclusive, None = until end)
            
        Example (single direction):
            >>> velocity_dir = torch.randn(512)
            >>> patcher.add_directional_ablation('encoder_output', direction=velocity_dir)
            
        Example (stored direction):
            >>> patcher.store_direction('velocity', velocity_dir)
            >>> patcher.add_directional_ablation('encoder_output', direction_name='velocity')
            
        Example (multiple directions):
            >>> dirs = [velocity_dir, position_dir, orientation_dir]
            >>> patcher.add_directional_ablation('encoder_output', directions=dirs)
            
        Example (step-gated ablation):
            >>> # Only ablate between steps 10 and 50
            >>> patcher.add_directional_ablation('encoder_output', 
            ...                                   direction_name='velocity',
            ...                                   start_step=10, end_step=50)
        """
        # Parse direction arguments
        dir_list = self._parse_direction_args(direction, direction_name,
                                              directions, direction_names,
                                              normalize)

        def ablation_fn(activation):
            # Check step bounds
            if start_step is not None and self.current_step < start_step:
                return activation
            if end_step is not None and self.current_step > end_step:
                return activation

            modified = activation.clone()

            # Remove each direction component sequentially
            for d in dir_list:
                # Move direction to same device as activation
                d = d.to(activation.device)
                modified = self._remove_direction_component(modified, d)

            return modified

        self.add_intervention(location, ablation_fn)
        step_str = self._format_step_bounds(start_step, end_step)
        if len(dir_list) == 1:
            print(f"  Ablating 1 directional component{step_str}")
        else:
            print(
                f"  Ablating {len(dir_list)} directional components{step_str}")

    def add_directional_scaling(self,
                                location: str,
                                scale: float,
                                direction: Optional[torch.Tensor] = None,
                                direction_name: Optional[str] = None,
                                directions: Optional[List[torch.Tensor]] = None,
                                direction_names: Optional[List[str]] = None,
                                normalize: bool = True,
                                start_step: Optional[int] = None,
                                end_step: Optional[int] = None):
        """
        Scale the component along direction(s) by a factor.
        
        Computes: x_new = x + (scale - 1) * proj_d(x)
        which is equivalent to: x_new = x_orthogonal + scale * proj_d(x)
        
        Args:
            location: Where to scale
            scale: Multiplicative factor for direction component(s)
                  - scale=0: equivalent to ablation
                  - scale=1: no change
                  - scale>1: amplify component
                  - scale<0: reverse component
            direction: Single direction vector of shape [features]
            direction_name: Name of stored direction to use
            directions: List of direction vectors (for multi-directional scaling)
            direction_names: List of stored direction names
            normalize: Whether to normalize direction(s) to unit length
            start_step: Step at which to start applying the intervention (inclusive, None = from beginning)
            end_step: Step at which to stop applying the intervention (inclusive, None = until end)
            
        Example:
            >>> # Amplify velocity component by 2x
            >>> patcher.add_directional_scaling('encoder_output', 
            ...                                   scale=2.0,
            ...                                   direction=velocity_dir)
            
            >>> # Reverse position component
            >>> patcher.add_directional_scaling('encoder_output',
            ...                                   scale=-1.0,
            ...                                   direction_name='position')
            
            >>> # Scale only during steps 20-100
            >>> patcher.add_directional_scaling('encoder_output',
            ...                                   scale=0.5,
            ...                                   direction_name='velocity',
            ...                                   start_step=20, end_step=100)
        """
        dir_list = self._parse_direction_args(direction, direction_name,
                                              directions, direction_names,
                                              normalize)

        def scaling_fn(activation):
            # Check step bounds
            if start_step is not None and self.current_step < start_step:
                return activation
            if end_step is not None and self.current_step > end_step:
                return activation

            modified = activation.clone()

            # Scale each direction component
            for d in dir_list:
                d = d.to(activation.device)
                projection = self._project_onto_direction(modified, d)
                # Remove original projection and add scaled version
                modified = modified - projection + scale * projection

            return modified

        self.add_intervention(location, scaling_fn)
        step_str = self._format_step_bounds(start_step, end_step)
        if len(dir_list) == 1:
            print(f"  Scaling 1 directional component by {scale}{step_str}")
        else:
            print(
                f"  Scaling {len(dir_list)} directional components by {scale}{step_str}"
            )

    def add_directional_addition(
            self,
            location: str,
            magnitude: float,
            direction: Optional[torch.Tensor] = None,
            direction_name: Optional[str] = None,
            directions: Optional[List[torch.Tensor]] = None,
            direction_names: Optional[List[str]] = None,
            magnitudes: Optional[List[float]] = None,
            normalize: bool = True,
            start_step: Optional[int] = None,
            end_step: Optional[int] = None):
        """
        Add (inject) a component along direction(s) to steer behavior.
        
        This is a key mechanistic interpretability technique for controlling
        model behavior. Unlike directional_scaling (which modifies existing
        components), this ADDS a new component regardless of what's already there.
        
        Computes: x_new = x + magnitude * d
        
        Args:
            location: Where to add the component ('encoder_output', etc.)
            magnitude: How much to add along the direction
                      - magnitude=0: no change
                      - magnitude>0: add in positive direction
                      - magnitude<0: add in negative direction
            direction: Single direction vector of shape [features]
            direction_name: Name of stored direction to use
            directions: List of direction vectors (for multi-directional addition)
            direction_names: List of stored direction names
            magnitudes: List of magnitudes (one per direction). If None, uses
                       the single 'magnitude' value for all directions.
            normalize: Whether to normalize direction(s) to unit length
            start_step: Step at which to start applying the intervention (inclusive, None = from beginning)
            end_step: Step at which to stop applying the intervention (inclusive, None = until end)
            
        Example - Controlling pole oscillation:
            >>> # Extract amplitude direction from episodes
            >>> amp_direction = extract_amplitude_direction(episodes)
            >>> patcher.store_direction('amplitude', amp_direction)
            >>> 
            >>> # Add positive amplitude to increase oscillation
            >>> patcher.add_directional_addition('encoder_output',
            ...                                   magnitude=2.0,
            ...                                   direction_name='amplitude')
            >>> 
            >>> # Or reduce oscillation
            >>> patcher.add_directional_addition('encoder_output',
            ...                                   magnitude=-1.0,
            ...                                   direction_name='amplitude')
            
        Example - Multi-directional steering:
            >>> # Simultaneously control amplitude and frequency
            >>> patcher.add_directional_addition('encoder_output',
            ...                                   magnitude=0.0,  # unused when magnitudes is provided
            ...                                   direction_names=['amplitude', 'frequency'],
            ...                                   magnitudes=[2.0, -0.5])
            
        Example - Step-gated addition:
            >>> # Only add component during steps 50-150
            >>> patcher.add_directional_addition('encoder_output',
            ...                                   magnitude=1.5,
            ...                                   direction_name='velocity',
            ...                                   start_step=50, end_step=150)
            
        Note: 
            This is different from directional_scaling:
            - directional_scaling: modifies the EXISTING component (x·d)
            - directional_addition: ADDS a NEW component independent of what exists
            
            For steering/control, directional_addition is usually more effective
            because it directly sets the feature level you want, rather than
            scaling whatever happens to be there naturally.
        """
        dir_list = self._parse_direction_args(direction, direction_name,
                                              directions, direction_names,
                                              normalize)

        # Handle magnitudes
        if magnitudes is None:
            mag_list = [magnitude] * len(dir_list)
        else:
            if len(magnitudes) != len(dir_list):
                raise ValueError(
                    f"Number of magnitudes ({len(magnitudes)}) must match "
                    f"number of directions ({len(dir_list)})")
            mag_list = magnitudes

        def addition_fn(activation):
            # Check step bounds
            if start_step is not None and self.current_step < start_step:
                return activation
            if end_step is not None and self.current_step > end_step:
                return activation

            modified = activation.clone()

            # Add each direction component
            for d, mag in zip(dir_list, mag_list):
                d = d.to(activation.device)
                # Simply add magnitude * direction
                modified = modified + mag * d

            return modified

        self.add_intervention(location, addition_fn)
        step_str = self._format_step_bounds(start_step, end_step)
        if len(dir_list) == 1:
            print(
                f"  Adding directional component with magnitude {magnitude:.3f}{step_str}"
            )
        else:
            mag_str = ", ".join([f"{m:.3f}" for m in mag_list])
            print(
                f"  Adding {len(dir_list)} directional components with magnitudes [{mag_str}]{step_str}"
            )

    # def add_directional_replacement(
    #         self,
    #         location: str,
    #         replacement_magnitude: float,
    #         direction: Optional[torch.Tensor] = None,
    #         direction_name: Optional[str] = None,
    #         directions: Optional[List[torch.Tensor]] = None,
    #         direction_names: Optional[List[str]] = None,
    #         replacement_magnitudes: Optional[List[float]] = None,
    #         normalize: bool = True):
    #     """
    #     Replace the projection along direction(s) with a specific magnitude.

    #     Computes: x_new = x - proj_d(x) + replacement_magnitude * d

    #     Args:
    #         location: Where to replace
    #         replacement_magnitude: Scalar magnitude to set projection to
    #         direction: Single direction vector of shape [features]
    #         direction_name: Name of stored direction to use
    #         directions: List of direction vectors
    #         direction_names: List of stored direction names
    #         replacement_magnitudes: List of magnitudes (one per direction, or single value for all)
    #         normalize: Whether to normalize direction(s) to unit length

    #     Example:
    #         >>> # Set velocity component to fixed value
    #         >>> patcher.add_directional_replacement('encoder_output',
    #         ...                                       replacement_magnitude=0.5,
    #         ...                                       direction_name='velocity')

    #         >>> # Set multiple components to different values
    #         >>> patcher.add_directional_replacement('encoder_output',
    #         ...                                       replacement_magnitude=0.0,
    #         ...                                       directions=[vel_dir, pos_dir],
    #         ...                                       replacement_magnitudes=[0.5, -0.3])
    #     """
    #     dir_list = self._parse_direction_args(direction, direction_name,
    #                                           directions, direction_names,
    #                                           normalize)

    #     # Handle replacement magnitudes
    #     if replacement_magnitudes is None:
    #         mag_list = [replacement_magnitude] * len(dir_list)
    #     else:
    #         if len(replacement_magnitudes) != len(dir_list):
    #             raise ValueError(
    #                 f"Number of replacement_magnitudes ({len(replacement_magnitudes)}) "
    #                 f"must match number of directions ({len(dir_list)})")
    #         mag_list = replacement_magnitudes

    #     def replacement_fn(activation):
    #         modified = activation.clone()

    #         # Replace each direction component
    #         for d, mag in zip(dir_list, mag_list):
    #             d = d.to(activation.device)
    #             projection = self._project_onto_direction(modified, d)
    #             # Remove original projection and add fixed magnitude
    #             modified = modified - projection + mag * d

    #         return modified

    #     self.add_intervention(location, replacement_fn)
    #     if len(dir_list) == 1:
    #         print(
    #             f"  Replacing 1 directional component with magnitude {replacement_magnitude}"
    #         )
    #     else:
    #         print(f"  Replacing {len(dir_list)} directional components")

    # def add_directional_clamp(self,
    #                           location: str,
    #                           min_magnitude: Optional[float] = None,
    #                           max_magnitude: Optional[float] = None,
    #                           direction: Optional[torch.Tensor] = None,
    #                           direction_name: Optional[str] = None,
    #                           directions: Optional[List[torch.Tensor]] = None,
    #                           direction_names: Optional[List[str]] = None,
    #                           normalize: bool = True):
    #     """
    #     Clamp the magnitude of projection along direction(s).

    #     Restricts the scalar projection (x · d) to be within [min_magnitude, max_magnitude].

    #     Args:
    #         location: Where to clamp
    #         min_magnitude: Minimum allowed magnitude (None = no lower bound)
    #         max_magnitude: Maximum allowed magnitude (None = no upper bound)
    #         direction: Single direction vector of shape [features]
    #         direction_name: Name of stored direction to use
    #         directions: List of direction vectors
    #         direction_names: List of stored direction names
    #         normalize: Whether to normalize direction(s) to unit length

    #     Example:
    #         >>> # Prevent velocity from exceeding certain bounds
    #         >>> patcher.add_directional_clamp('encoder_output',
    #         ...                                 min_magnitude=-1.0,
    #         ...                                 max_magnitude=1.0,
    #         ...                                 direction_name='velocity')
    #     """
    #     if min_magnitude is None and max_magnitude is None:
    #         raise ValueError(
    #             "At least one of min_magnitude or max_magnitude must be specified"
    #         )

    #     dir_list = self._parse_direction_args(direction, direction_name,
    #                                           directions, direction_names,
    #                                           normalize)

    #     def clamp_fn(activation):
    #         modified = activation.clone()

    #         for d in dir_list:
    #             d = d.to(activation.device)

    #             # Compute scalar projection
    #             scalar_proj = torch.sum(modified * d, dim=-1, keepdim=True)

    #             # Clamp the scalar projection
    #             clamped_scalar = scalar_proj
    #             if min_magnitude is not None:
    #                 clamped_scalar = torch.maximum(
    #                     clamped_scalar,
    #                     torch.tensor(min_magnitude, device=activation.device))
    #             if max_magnitude is not None:
    #                 clamped_scalar = torch.minimum(
    #                     clamped_scalar,
    #                     torch.tensor(max_magnitude, device=activation.device))

    #             # Replace projection with clamped version
    #             original_proj = scalar_proj * d
    #             clamped_proj = clamped_scalar * d
    #             modified = modified - original_proj + clamped_proj

    #         return modified

    #     self.add_intervention(location, clamp_fn)
    #     clamp_str = f"[{min_magnitude}, {max_magnitude}]"
    #     if len(dir_list) == 1:
    #         print(f"  Clamping 1 directional component to {clamp_str}")
    #     else:
    #         print(
    #             f"  Clamping {len(dir_list)} directional components to {clamp_str}"
    #         )

    def _format_step_bounds(self, start_step: Optional[int],
                            end_step: Optional[int]) -> str:
        """Format step bounds for printing."""
        if start_step is None and end_step is None:
            return ""
        elif start_step is not None and end_step is not None:
            return f" (steps {start_step}-{end_step})"
        elif start_step is not None:
            return f" (from step {start_step})"
        else:
            return f" (until step {end_step})"

    def _parse_direction_args(self, direction: Optional[torch.Tensor],
                              direction_name: Optional[str],
                              directions: Optional[List[torch.Tensor]],
                              direction_names: Optional[List[str]],
                              normalize: bool) -> List[torch.Tensor]:
        """
        Parse and validate direction arguments, returning a list of direction tensors.
        
        Args:
            direction: Single direction vector
            direction_name: Name of stored direction
            directions: List of direction vectors
            direction_names: List of stored direction names
            normalize: Whether to normalize directions
            
        Returns:
            List of direction tensors (normalized if requested)
        """
        dir_list = []

        # Count how many direction specifications were provided
        specs_provided = sum([
            direction is not None, direction_name is not None, directions
            is not None, direction_names is not None
        ])

        if specs_provided == 0:
            raise ValueError(
                "Must provide one of: direction, direction_name, directions, or direction_names"
            )
        elif specs_provided > 1:
            raise ValueError(
                "Must provide exactly one of: direction, direction_name, directions, or direction_names"
            )

        # Parse single direction
        if direction is not None:
            if normalize:
                direction = self._normalize_direction(direction)
            dir_list = [direction]

        # Parse single direction name
        elif direction_name is not None:
            if direction_name not in self.directions:
                raise ValueError(
                    f"Direction '{direction_name}' not found. Use store_direction() first."
                )
            dir_list = [self.directions[direction_name]]

        # Parse multiple directions
        elif directions is not None:
            for d in directions:
                if normalize:
                    d = self._normalize_direction(d)
                dir_list.append(d)

        # Parse multiple direction names
        elif direction_names is not None:
            for name in direction_names:
                if name not in self.directions:
                    raise ValueError(
                        f"Direction '{name}' not found. Use store_direction() first."
                    )
                dir_list.append(self.directions[name])

        return dir_list

    # ========================================================================
    # END DIRECTIONAL INTERVENTION METHODS
    # ========================================================================

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

    def set_step(self, step: int):
        """
        Set the current step counter for step-gated interventions.
        
        Call this at each step of your rollout to update the step counter.
        Interventions with start_step/end_step will only apply when
        current_step is within [start_step, end_step].
        
        Args:
            step: Current step number (0-indexed)
            
        Example:
            >>> for step in range(max_steps):
            ...     patcher.set_step(step)
            ...     action = agent.act(obs)  # Interventions will check step bounds
            ...     obs, reward, done, info = env.step(action)
        """
        self.current_step = step

    def reset_step(self):
        """Reset the step counter to 0. Call at the start of each episode."""
        self.current_step = 0

    def increment_step(self):
        """Increment the step counter by 1. Alternative to set_step()."""
        self.current_step += 1

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
        """Print summary of current interventions and stored directions."""
        print("\n" + "=" * 60)
        print("ActivationPatcher Summary")
        print("=" * 60)
        print(f"Enabled: {self.enabled}")
        print(f"Current step: {self.current_step}")
        print(f"Number of interventions: {len(self.interventions)}")
        print(f"Number of stored directions: {len(self.directions)}")

        if self.interventions:
            print("\nActive interventions:")
            for location in self.interventions:
                print(f"  - {location}")
        else:
            print("\nNo active interventions")

        if self.directions:
            print("\nStored directions:")
            for name, direction in self.directions.items():
                print(
                    f"  - '{name}': shape {tuple(direction.shape)}, norm={torch.norm(direction).item():.4f}"
                )
        else:
            print("\nNo stored directions")

        print("=" * 60 + "\n")
