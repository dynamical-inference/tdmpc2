"""
Closed-loop trajectory controller for TD-MPC2 via latent steering.

This module provides TrajectoryController, which modifies agent activations
at each step to make the agent follow a desired trajectory.

The controller works by:
1. At each step, comparing current state to target trajectory
2. Computing a steering signal (error or gradient)
3. Injecting this signal into the latent space via ActivationPatcher

Supported control modes:
- 'proportional': Simple P-controller on trajectory error
- 'pid': Full PID controller for smoother tracking
- 'gradient': Gradient-based steering (requires differentiable decoder)
- 'magnitude_only': Fixed steering direction, only magnitude varies with error (closed-loop)
- 'open_loop': Pre-specified magnitude sequence, no feedback (open-loop)

Example usage:
    from common.trajectory_controller import TrajectoryController
    from common.activation_patcher import ActivationPatcher
    
    # Define target trajectory (e.g., cart position over 500 steps)
    target_trajectory = torch.zeros(500, 5)  # [cart_pos, cos_angle, sin_angle, cart_vel, pole_vel]
    target_trajectory[:, 0] = torch.linspace(-0.5, 0.5, 500)  # Move cart from -0.5 to 0.5
    target_trajectory[:, 1] = 1.0  # cos(0) = 1 (upright)
    target_trajectory[:, 2] = 0.0  # sin(0) = 0 (upright)
    
    # Create patcher and controller
    patcher = ActivationPatcher(agent.model, 
                                 renormalize_after_intervention=True,
                                 normalization_method='simnorm',
                                 simnorm_dim=8)
    
    # Option 1: If you have learned steering directions (from probing/PCA)
    controller = TrajectoryController(
        patcher=patcher,
        target_trajectory=target_trajectory,
        steering_directions={'cart_pos': cart_pos_direction, 'angle': angle_direction},
        feature_indices={'cart_pos': 0, 'angle': (1, 2)},  # Which obs dims to track
        control_mode='proportional',
        gain=1.0,
    )
    
    # Option 2: If you have a decoder z -> obs
    controller = TrajectoryController(
        patcher=patcher,
        target_trajectory=target_trajectory,
        decoder=my_decoder,  # Function: z -> predicted_obs
        control_mode='gradient',
        gain=0.1,
    )
    
    # Option 3: Fixed steering direction, only magnitude changes (closed-loop)
    controller = TrajectoryController(
        patcher=patcher,
        target_trajectory=target_trajectory,
        fixed_steering_direction=my_steering_vector,  # Shape: (latent_dim,)
        feature_indices={'cart_pos': 0},  # Feature(s) to compute error from
        control_mode='magnitude_only',
        gain=1.0,
    )
    
    # Option 4: Pre-specified magnitude sequence (open-loop, no feedback)
    magnitudes = torch.linspace(0, 2.0, 500)  # Ramp up steering over episode
    controller = TrajectoryController(
        patcher=patcher,
        target_trajectory=target_trajectory,  # Can be dummy, not used for control
        fixed_steering_direction=my_steering_vector,  # Shape: (latent_dim,)
        prescribed_magnitudes=magnitudes,  # Shape: (num_steps,)
        control_mode='open_loop',
    )
    
    # Run episode with trajectory following
    obs = env.reset()
    for step in range(500):
        # Update controller with current observation
        controller.update(obs, step)
        
        # Agent acts with modified latent
        action = agent.act(obs, t0=(step==0), eval_mode=True)
        obs, reward, done, info = env.step(action)
        
    controller.reset()
"""

import torch
from typing import Dict, List, Optional, Union, Callable, Tuple
import warnings
import numpy as np


class TrajectoryController:
    """
    Closed-loop controller that steers TD-MPC2 to follow a target trajectory.
    
    Works by injecting steering signals into the latent space at each step,
    computed as a function of the tracking error (target - current).
    """

    def __init__(
            self,
            patcher: 'ActivationPatcher',
            target_trajectory: torch.Tensor,
            steering_directions: Optional[Dict[str, torch.Tensor]] = None,
            feature_indices: Optional[Dict[str, Union[int, Tuple[int,
                                                                 ...]]]] = None,
            decoder: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
            fixed_steering_direction: Optional[torch.Tensor] = None,
            prescribed_magnitudes: Optional[torch.Tensor] = None,
            control_mode: str = 'proportional',
            gain: Union[float, Dict[str, float]] = 1.0,
            ki: Union[float, Dict[str, float]] = 0.0,  # Integral gain for PID
            kd: Union[float, Dict[str, float]] = 0.0,  # Derivative gain for PID
            max_steering_magnitude: float = 10.0,
            intervention_location: str = 'encoder_output',
            observation_to_features: Optional[Callable[[torch.Tensor],
                                                       torch.Tensor]] = None,
            error_aggregation:
        str = 'sum',  # For magnitude_only: 'sum', 'mean', 'norm'
    ):
        """
        Initialize trajectory controller.
        
        Args:
            patcher: ActivationPatcher instance attached to the agent's model
            target_trajectory: Target trajectory tensor of shape (num_steps, num_features)
                              Features should match what you want to track (e.g., obs dims)
            steering_directions: Dict mapping feature names to direction vectors in latent space
                                Required for 'proportional' and 'pid' modes
            feature_indices: Dict mapping feature names to indices in the observation/target
                            e.g., {'cart_pos': 0, 'angle': (1, 2)} for cartpole
            decoder: Function that maps latent z to predicted observations
                    Required for 'gradient' mode
            fixed_steering_direction: Fixed steering vector in latent space.
                    Required for 'magnitude_only' and 'open_loop' modes. Shape: (latent_dim,)
            prescribed_magnitudes: Pre-specified magnitude sequence for 'open_loop' mode.
                    Shape: (num_steps,). Steering at step t = magnitudes[t] * fixed_direction
            control_mode: One of:
                - 'proportional': Steering = gain * error * direction[feature]
                - 'pid': Full PID controller
                - 'gradient': Gradient-based steering through decoder
                - 'magnitude_only': Fixed direction, magnitude = gain * aggregated_error (closed-loop)
                - 'open_loop': Fixed direction, magnitude from prescribed_magnitudes[t] (no feedback)
            gain: Proportional gain (Kp). Can be float or dict per feature.
            error_aggregation: How to aggregate errors for 'magnitude_only' mode:
                - 'sum': Sum of all feature errors (signed)
                - 'mean': Mean of all feature errors (signed)
                - 'norm': L2 norm of error vector (always positive)
            ki: Integral gain (Ki) for PID mode
            kd: Derivative gain (Kd) for PID mode
            max_steering_magnitude: Maximum steering magnitude (for safety)
            intervention_location: Where to inject steering (default: 'encoder_output')
            observation_to_features: Optional function to convert raw obs to features
                                    If None, uses identity (assumes obs already has target features)
        """
        self.patcher = patcher
        self.target_trajectory = target_trajectory
        self.steering_directions = steering_directions or {}
        self.feature_indices = feature_indices or {}
        self.decoder = decoder
        self.fixed_steering_direction = fixed_steering_direction
        self.prescribed_magnitudes = prescribed_magnitudes
        self.control_mode = control_mode
        self.max_steering_magnitude = max_steering_magnitude
        self.intervention_location = intervention_location
        self.observation_to_features = observation_to_features
        self.error_aggregation = error_aggregation

        # Convert gains to dicts if scalars
        self.kp = self._to_gain_dict(gain, 'kp')
        self.ki = self._to_gain_dict(ki, 'ki')
        self.kd = self._to_gain_dict(kd, 'kd')

        # Validate configuration
        self._validate_config()

        # Controller state
        self.current_step = 0
        self.integral_error = {}  # For PID
        self.prev_error = {}  # For PID
        self.last_observation = None
        self.last_steering = None

        # Register the steering intervention
        self._register_intervention()

        print("Normalizing fixed steering direction")
        self.fixed_steering_direction = self.fixed_steering_direction / torch.norm(
            self.fixed_steering_direction)

        print(f"TrajectoryController initialized:")
        print(f"  Mode: {control_mode}")
        print(f"  Target trajectory shape: {tuple(target_trajectory.shape)}")
        print(f"  Features: {list(self.feature_indices.keys())}")
        print(f"  Gain (Kp): {self.kp}")
        if control_mode == 'magnitude_only' and self.fixed_steering_direction is not None:
            print(
                f"  Fixed direction shape: {tuple(self.fixed_steering_direction.shape)}"
            )
            print(f"  Error aggregation: {self.error_aggregation}")
        if control_mode == 'open_loop' and self.fixed_steering_direction is not None:
            print(
                f"  Fixed direction shape: {tuple(self.fixed_steering_direction.shape)}"
            )
            if self.prescribed_magnitudes is not None:
                print(
                    f"  Prescribed magnitudes shape: {tuple(self.prescribed_magnitudes.shape)}"
                )
                print(
                    f"  Magnitude range: [{self.prescribed_magnitudes.min():.3f}, {self.prescribed_magnitudes.max():.3f}]"
                )

    def _to_gain_dict(self, gain: Union[float, Dict[str, float]],
                      name: str) -> Dict[str, float]:
        """Convert scalar gain to dict, or validate dict gain."""
        if isinstance(gain, (int, float)):
            return {feat: float(gain) for feat in self.feature_indices}
        return gain

    def _validate_config(self):
        """Validate controller configuration."""
        if self.control_mode in ['proportional', 'pid']:
            if not self.steering_directions:
                raise ValueError(
                    f"steering_directions required for '{self.control_mode}' mode. "
                    "Provide direction vectors from probing/PCA.")
            if not self.feature_indices:
                raise ValueError(
                    "feature_indices required to map observation dims to features."
                )
            # Check all features have directions
            for feat in self.feature_indices:
                if feat not in self.steering_directions:
                    raise ValueError(
                        f"Missing steering direction for feature '{feat}'. "
                        f"Available: {list(self.steering_directions.keys())}")

        elif self.control_mode == 'gradient':
            if self.decoder is None:
                raise ValueError("decoder required for 'gradient' mode. "
                                 "Provide a function z -> predicted_obs.")

        elif self.control_mode == 'magnitude_only':
            if self.fixed_steering_direction is None:
                raise ValueError(
                    "fixed_steering_direction required for 'magnitude_only' mode. "
                    "Provide a steering vector of shape (latent_dim,).")
            if not self.feature_indices:
                raise ValueError(
                    "feature_indices required to compute tracking error.")
            if self.error_aggregation not in ['sum', 'mean', 'norm']:
                raise ValueError(
                    f"Unknown error_aggregation: '{self.error_aggregation}'. "
                    "Use 'sum', 'mean', or 'norm'.")

        elif self.control_mode == 'open_loop':
            if self.fixed_steering_direction is None:
                raise ValueError(
                    "fixed_steering_direction required for 'open_loop' mode. "
                    "Provide a steering vector of shape (latent_dim,).")
            if self.prescribed_magnitudes is None:
                raise ValueError(
                    "prescribed_magnitudes required for 'open_loop' mode. "
                    "Provide a tensor of shape (num_steps,) with magnitude for each timestep."
                )

        else:
            raise ValueError(
                f"Unknown control_mode: '{self.control_mode}'. "
                "Use 'proportional', 'pid', 'gradient', 'magnitude_only', or 'open_loop'."
            )

    def _register_intervention(self):
        """Register the steering intervention with the patcher."""

        # We use a custom intervention that computes steering dynamically
        def steering_intervention(activation: torch.Tensor) -> torch.Tensor:
            assert self.last_steering is not None

            # Add steering to activation
            steering = self.last_steering.to(activation.device)

            # Handle different activation shapes
            if activation.dim() == 2:  # [batch, latent_dim]
                steering = steering.unsqueeze(0).expand(activation.shape[0], -1)
            elif activation.dim() == 1:  # [latent_dim]
                pass  # steering is already [latent_dim]

            return activation + steering

        self.patcher.add_intervention(
            self.intervention_location,
            steering_intervention,
            start_step=0,
            end_step=None,
        )

    def _extract_features(self,
                          observation: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Extract tracked features from observation."""
        if self.observation_to_features is not None:
            obs = self.observation_to_features(observation)
        else:
            obs = observation

        features = {}
        for feat_name, indices in self.feature_indices.items():
            if isinstance(indices, int):
                features[feat_name] = obs[..., indices]
            elif isinstance(indices, (tuple, list)):
                features[feat_name] = obs[..., list(indices)]
            else:
                features[feat_name] = obs[..., indices]

        return features

    def _get_target_features(self, step: int) -> Dict[str, torch.Tensor]:
        """Get target features for the current step."""
        # Clamp step to valid range
        step = min(step, self.target_trajectory.shape[0] - 1)
        target = self.target_trajectory[step]

        features = {}
        for feat_name, indices in self.feature_indices.items():
            if isinstance(indices, int):
                features[feat_name] = target[indices]
            elif isinstance(indices, (tuple, list)):
                features[feat_name] = target[list(indices)]
            else:
                features[feat_name] = target[indices]

        return features

    def _compute_error(
        self,
        current_features: Dict[str, torch.Tensor],
        target_features: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """Compute tracking error for each feature."""
        errors = {}
        for feat_name in self.feature_indices:
            current = current_features[feat_name]
            target = target_features[feat_name]

            # Handle angle features (cos, sin) specially
            if isinstance(current, torch.Tensor) and current.numel() == 2:
                # For angles represented as (cos, sin), compute angular error
                # This handles the wrap-around correctly
                current_angle = torch.atan2(current[1], current[0])
                target_angle = torch.atan2(target[1], target[0])
                # Angular error (handles wrap-around)
                error = torch.atan2(torch.sin(target_angle - current_angle),
                                    torch.cos(target_angle - current_angle))
                errors[feat_name] = error
            else:
                errors[feat_name] = target - current

        return errors

    def _compute_proportional_steering(
        self,
        errors: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Compute steering using proportional control."""
        steering = torch.zeros_like(list(self.steering_directions.values())[0])

        for feat_name, error in errors.items():
            direction = self.steering_directions[feat_name]
            kp = self.kp.get(feat_name, 1.0)

            # Scalar error times direction
            if isinstance(error, torch.Tensor):
                error_scalar = error.mean() if error.numel() > 1 else error
            else:
                error_scalar = error

            steering = steering + kp * error_scalar * direction.to(
                steering.device)

        return steering

    def _compute_pid_steering(
        self,
        errors: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Compute steering using PID control."""
        steering = torch.zeros_like(list(self.steering_directions.values())[0])

        for feat_name, error in errors.items():
            direction = self.steering_directions[feat_name]
            kp = self.kp.get(feat_name, 1.0)
            ki = self.ki.get(feat_name, 0.0)
            kd = self.kd.get(feat_name, 0.0)

            # Convert to scalar
            if isinstance(error, torch.Tensor):
                error_scalar = error.mean() if error.numel(
                ) > 1 else error.item()
            else:
                error_scalar = error

            # Initialize state if needed
            if feat_name not in self.integral_error:
                self.integral_error[feat_name] = 0.0
                self.prev_error[feat_name] = error_scalar

            # PID computation
            p_term = kp * error_scalar
            self.integral_error[feat_name] += error_scalar
            i_term = ki * self.integral_error[feat_name]
            d_term = kd * (error_scalar - self.prev_error[feat_name])
            self.prev_error[feat_name] = error_scalar

            pid_output = p_term + i_term + d_term
            steering = steering + pid_output * direction.to(steering.device)

        return steering

    def _compute_magnitude_only_steering(
        self,
        errors: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Compute steering using fixed direction with variable magnitude.
        
        The steering direction remains constant throughout the episode.
        Only the magnitude (scalar multiplier) changes based on tracking error.
        """
        # Aggregate errors into a single scalar
        error_values = []
        for feat_name, error in errors.items():
            if isinstance(error, torch.Tensor):
                error_val = error.mean() if error.numel() > 1 else error
            else:
                error_val = torch.tensor(error)
            error_values.append(error_val)

        # Stack into tensor for aggregation
        error_tensor = torch.stack([
            e if isinstance(e, torch.Tensor) else torch.tensor(e)
            for e in error_values
        ])

        # Aggregate based on chosen method
        if self.error_aggregation == 'sum':
            magnitude = error_tensor.sum()
        elif self.error_aggregation == 'mean':
            magnitude = error_tensor.mean()
        elif self.error_aggregation == 'norm':
            magnitude = torch.norm(error_tensor)
        else:
            magnitude = error_tensor.sum()

        # Get gain (use first value if dict, since direction is fixed)
        if isinstance(self.kp, dict):
            gain = list(self.kp.values())[0] if self.kp else 1.0
        else:
            gain = self.kp

        # Scale the fixed direction by gain * magnitude
        assert self.fixed_steering_direction is not None  # Validated in __init__
        steering = gain * magnitude * self.fixed_steering_direction.to(
            error_tensor.device)

        return steering

    def _compute_open_loop_steering(self, step: int) -> torch.Tensor:
        """Compute steering using pre-specified magnitude sequence (no feedback).
        
        The steering direction is fixed and the magnitude at each timestep
        is read directly from prescribed_magnitudes[step].
        """
        assert self.fixed_steering_direction is not None  # Validated in __init__
        assert self.prescribed_magnitudes is not None  # Validated in __init__

        # Clamp step to valid range (prevent index out of bounds)
        step = min(step, len(self.prescribed_magnitudes) - 1)

        # Get magnitude for this timestep
        magnitude = self.prescribed_magnitudes[step]

        # Scale the fixed direction
        steering = magnitude * self.fixed_steering_direction.to(
            self.prescribed_magnitudes.device)

        return steering

    # def _compute_gradient_steering(
    #     self,
    #     observation: torch.Tensor,
    #     target_features: Dict[str, torch.Tensor],
    #     latent: Optional[torch.Tensor] = None,
    # ) -> torch.Tensor:
    #     """Compute steering using gradient through decoder."""
    #     if latent is None:
    #         # Get the last encoded latent from patcher
    #         latent = self.patcher.get_activation(self.intervention_location)
    #         if latent is None:
    #             warnings.warn(
    #                 "No latent available for gradient steering. Skipping.")
    #             return torch.zeros(512)  # Return zero steering

    #     # Enable gradients for latent
    #     latent = latent.detach().clone().requires_grad_(True)

    #     # Decode latent
    #     predicted_obs = self.decoder(latent)

    #     # Compute loss (MSE to target)
    #     target_tensor = torch.cat([
    #         t.flatten() if isinstance(t, torch.Tensor) else torch.tensor([t])
    #         for t in target_features.values()
    #     ])

    #     # Extract corresponding predicted features
    #     pred_features = []
    #     for feat_name, indices in self.feature_indices.items():
    #         if isinstance(indices, int):
    #             pred_features.append(predicted_obs[..., indices].flatten())
    #         else:
    #             pred_features.append(predicted_obs[...,
    #                                                list(indices)].flatten())
    #     pred_tensor = torch.cat(pred_features)

    #     loss = torch.nn.functional.mse_loss(
    #         pred_tensor, target_tensor.to(pred_tensor.device))

    #     # Backprop to get gradient
    #     loss.backward()

    #     # Gradient gives direction to INCREASE loss, we want to DECREASE it
    #     steering = -self.kp.get(list(self.feature_indices.keys())[0],
    #                             1.0) * latent.grad

    #     return steering.detach()

    def update(
        self,
        observation: torch.Tensor,
        step: Optional[int] = None,
        latent: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Update controller with current observation and compute steering.
        
        Call this BEFORE agent.act() at each step.
        
        Args:
            observation: Current observation from environment
            step: Current timestep (if None, uses internal counter)
            latent: Optional current latent state (for gradient mode)
            
        Returns:
            Dict with debugging info (errors, steering magnitude, etc.)
        """
        if step is not None:
            self.current_step = step

        # Convert to tensor if needed
        if not isinstance(observation, torch.Tensor):
            observation = torch.tensor(observation, dtype=torch.float32)

        # Extract current features
        current_features = self._extract_features(observation)
        target_features = self._get_target_features(self.current_step)

        # Compute errors
        errors = self._compute_error(current_features, target_features)

        # Compute steering based on mode
        if self.control_mode == 'proportional':
            steering = self._compute_proportional_steering(errors)
        elif self.control_mode == 'pid':
            steering = self._compute_pid_steering(errors)
        elif self.control_mode == 'magnitude_only':
            steering = self._compute_magnitude_only_steering(errors)
        elif self.control_mode == 'open_loop':
            steering = self._compute_open_loop_steering(self.current_step)
        # elif self.control_mode == 'gradient':
        #     steering = self._compute_gradient_steering(observation,
        #                                                target_features, latent)
        else:
            raise ValueError(f"Unknown control_mode: {self.control_mode}")

        # Clamp steering magnitude
        # steering_norm = torch.norm(steering)
        # if steering_norm > self.max_steering_magnitude:
        #     steering = steering * (self.max_steering_magnitude / steering_norm)

        # Store for intervention
        self.last_steering = steering
        self.last_observation = observation

        # Update patcher timestep
        self.patcher.current_step = self.current_step

        # Increment internal counter
        self.current_step += 1

        # Return debug info
        result = {
            'errors': errors,
            #'steering_norm': steering_norm.item(),
            'current_features': current_features,
            'target_features': target_features,
        }
        # Add prescribed magnitude info for open_loop mode
        if self.control_mode == 'open_loop' and self.prescribed_magnitudes is not None:
            step_idx = min(self.current_step - 1,
                           len(self.prescribed_magnitudes) - 1)
            result['prescribed_magnitude'] = self.prescribed_magnitudes[
                step_idx].item()
        return result

    def reset(self):
        """Reset controller state for new episode."""
        self.current_step = 0
        self.integral_error = {}
        self.prev_error = {}
        self.last_observation = None
        self.last_steering = None
        self.patcher.reset_timestep()
        print("TrajectoryController reset")

    def set_target_trajectory(self, trajectory: torch.Tensor):
        """Update the target trajectory."""
        self.target_trajectory = trajectory
        print(f"Updated target trajectory: shape {tuple(trajectory.shape)}")

    def set_prescribed_magnitudes(self, magnitudes: torch.Tensor):
        """Update the prescribed magnitude sequence for open_loop mode."""
        self.prescribed_magnitudes = magnitudes
        print(
            f"Updated prescribed magnitudes: shape {tuple(magnitudes.shape)}, "
            f"range [{magnitudes.min():.3f}, {magnitudes.max():.3f}]")

    def get_tracking_error(self) -> Dict[str, float]:
        """Get current tracking errors (call after update)."""
        if self.last_observation is None:
            return {}

        current_features = self._extract_features(self.last_observation)
        target_features = self._get_target_features(self.current_step - 1)
        errors = self._compute_error(current_features, target_features)

        return {
            k: v.item() if isinstance(v, torch.Tensor) else v
            for k, v in errors.items()
        }


# class AdaptiveTrajectoryController(TrajectoryController):
#     """
#     Trajectory controller with adaptive gain based on tracking performance.

#     Increases gain when tracking error is high, decreases when tracking well.
#     """

#     def __init__(self,
#                  *args,
#                  adaptation_rate: float = 0.01,
#                  min_gain: float = 0.1,
#                  max_gain: float = 10.0,
#                  target_error: float = 0.1,
#                  **kwargs):
#         super().__init__(*args, **kwargs)
#         self.adaptation_rate = adaptation_rate
#         self.min_gain = min_gain
#         self.max_gain = max_gain
#         self.target_error = target_error
#         self.error_history = []

#     def update(self, observation, step=None, latent=None):
#         result = super().update(observation, step, latent)

#         # Adapt gains based on error
#         total_error = sum(
#             abs(e.item() if isinstance(e, torch.Tensor) else e)
#             for e in result['errors'].values())
#         self.error_history.append(total_error)

#         # Simple adaptive law: increase gain if error > target, decrease otherwise
#         for feat_name in self.kp:
#             if total_error > self.target_error:
#                 self.kp[feat_name] = min(
#                     self.kp[feat_name] * (1 + self.adaptation_rate),
#                     self.max_gain)
#             else:
#                 self.kp[feat_name] = max(
#                     self.kp[feat_name] * (1 - self.adaptation_rate),
#                     self.min_gain)

#         result['adapted_gains'] = self.kp.copy()
#         return result

# def create_cartpole_trajectory(
#     num_steps: int = 500,
#     cart_trajectory: Optional[torch.Tensor] = None,
#     angle_trajectory: Optional[torch.Tensor] = None,
#     keep_upright: bool = True,
# ) -> torch.Tensor:
#     """
#     Create a target trajectory for cartpole swingup.

#     Args:
#         num_steps: Number of timesteps
#         cart_trajectory: Optional cart position trajectory (num_steps,)
#         angle_trajectory: Optional pole angle trajectory in radians (num_steps,)
#         keep_upright: If True and angle_trajectory is None, keep pole upright

#     Returns:
#         Trajectory tensor of shape (num_steps, 5) for cartpole observations:
#         [cart_pos, cos(angle), sin(angle), cart_vel, pole_vel]
#     """
#     trajectory = torch.zeros(num_steps, 5)

#     # Cart position
#     if cart_trajectory is not None:
#         trajectory[:, 0] = cart_trajectory
#     else:
#         trajectory[:, 0] = 0.0  # Keep centered

#     # Pole angle (as cos, sin)
#     if angle_trajectory is not None:
#         trajectory[:, 1] = torch.cos(angle_trajectory)
#         trajectory[:, 2] = torch.sin(angle_trajectory)
#     elif keep_upright:
#         trajectory[:, 1] = 1.0  # cos(0)
#         trajectory[:, 2] = 0.0  # sin(0)

#     # Velocities (default to zero)
#     trajectory[:, 3] = 0.0  # cart velocity
#     trajectory[:, 4] = 0.0  # pole velocity

#     return trajectory
