from __future__ import annotations
"""
Black-box latent-space causal gradient estimation utilities.

These helpers support multiple modes for estimating how an arbitrary scalar 
metric J changes with respect to the latent state:

1. Environment rollouts with finite differences (SPSA)
2. Direct latent objectives with finite differences (SPSA)
3. Direct latent objectives with backpropagation (exact gradients)

Typical usage - Environment rollouts:

	normalizer = LatentNormalizer(group_dim=cfg.simnorm_dim)
	rollout_runner = RolloutRunner(
		planner=latent_planner_callable,
		env=env,
		encoder=encoder,
		objective=my_metric_function,
		normalizer=normalizer,
		horizon=cfg.episode_length,
	)
	probe = FiniteDifferenceProbe(
		rollout_runner=rollout_runner,
		normalizer=normalizer,
		num_directions=32,
		perturbation_scale=1e-2,
	)
	result = probe.estimate(z, initial_obs, initial_state)
	print(result.gradient)

Typical usage - Direct latent objective with backprop:

	# Define decoder and objective
	decoder = lambda z: z[0]  # Extract first component (e.g., cart position)
	objective = lambda pos: -torch.abs(pos)  # Minimize distance from 0
	
	# Create runner and probe
	latent_runner = LatentObjectiveRunner(
		objective=objective,
		decoder=decoder,
		track_gradients=True,
	)
	probe = BackpropGradientProbe(
		objective_runner=latent_runner,
		normalizer=normalizer,
	)
	
	# Compute exact gradient
	z = z.requires_grad_(True)
	gradient, result = probe.estimate(z)
	print(gradient)

"""

from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple
from common.seed import set_seed
import torch
from torch import Tensor

TensorLike = Tensor


@dataclass
class RolloutTrajectory:
    observations: Tensor
    latents: List[Tensor]
    actions: Tensor
    aux: Sequence[Any]


@dataclass
class RolloutResult:
    """
	Result of a rollout consisting of the trajectory and the scalar score.
	"""

    trajectory: RolloutTrajectory
    score: Tensor


@dataclass
class CausalGradientResult:
    """
	Aggregate output of a finite-difference causal gradient estimate.
	"""

    gradient: Tensor
    baseline: RolloutResult
    directions: Tensor
    perturbation_scale: float
    positive_rollouts: Sequence[RolloutResult]
    negative_rollouts: Sequence[RolloutResult]


@dataclass
class LatentObjectiveResult:
    """
	Result of evaluating an objective directly on a latent state.
	"""

    latent: Tensor
    decoded: Optional[Tensor]  # Decoder output if decoder was used
    score: Tensor


class LatentNormalizer:
    """
	Utility to keep latent vectors on the SimNorm manifold.

	The latent vector is partitioned into groups of size `group_dim`. Each group
	is normalized with a softmax (optionally using an inverse temperature),
	and the groups are concatenated back together.
	"""

    def __init__(self, group_dim: int, temperature: float = 1.0):
        if group_dim <= 0:
            raise ValueError("group_dim must be positive.")
        if temperature <= 0:
            raise ValueError("temperature must be positive.")
        self.group_dim = group_dim
        self.temperature = temperature

    def normalize(self, z: TensorLike) -> TensorLike:
        """


        Project a latent vector back onto the simplex-concatenated manifold.
        """

        if z.ndim == 1:
            z = z.unsqueeze(0)
            squeeze = True
        else:
            squeeze = False
        if z.shape[-1] % self.group_dim != 0:
            raise ValueError(
                f"Latent dimension {z.shape[-1]} is not divisible by group_dim={self.group_dim}."
            )
        grouped = z.view(*z.shape[:-1], -1, self.group_dim)
        grouped = grouped / self.temperature
        grouped = torch.softmax(grouped, dim=-1)
        normalized = grouped.view(*z.shape)
        if squeeze:
            normalized = normalized.squeeze(0)
        return normalized

    def project(self, z: TensorLike, delta: TensorLike) -> TensorLike:
        """
    	Add a perturbation and re-normalize the latent representation.
    	"""
        return self.normalize(z + delta)

    def project_no_norm(self, z: TensorLike, delta: TensorLike) -> TensorLike:
        """
		Add a perturbation and do not re-normalize the latent representation.
		"""
        return z + delta


class RandomDirectionSampler:
    """
	Samples isotropic random directions in latent space for SPSA estimates.
	
	If a generator with a seed is provided, the sampler will cache and return 
	the same directions on every call to sample() for reproducibility.
	"""

    def __init__(self,
                 dim: int,
                 num_directions: int,
                 generator: Optional[torch.Generator] = None):
        if dim <= 0:
            raise ValueError("dim must be positive.")
        if num_directions <= 0:
            raise ValueError("num_directions must be positive.")
        self.dim = dim
        self.num_directions = num_directions
        self._generator = generator
        self._cached_directions: Optional[Tensor] = None

    def sample(self, *, device: torch.device, dtype: torch.dtype) -> Tensor:
        """
		Draw `num_directions` normalized direction vectors.
		
		If a generator was provided in __init__, this will cache and return the same
		directions every time for reproducibility.
		"""

        # NOTE(R): I need to improve this... maybe generator here is not the best idea.
        # If we have a generator and cached directions, return the cached version
        if self._generator is not None and self._cached_directions is not None:
            # Return cached directions (move to requested device/dtype if needed)
            return self._cached_directions.to(device=device, dtype=dtype)

        # Ensure generator device matches the requested device for reproducibility
        if self._generator is not None:
            gen_device = self._generator.device
            req_device = torch.device(device)
            assert gen_device == req_device, (
                f"Generator device ({gen_device}) must match sampling device ({req_device}). "
                f"Create the generator on the same device: torch.Generator(device='{req_device}').manual_seed(seed)"
            )

        # Sample new directions
        dirs = torch.randn(self.num_directions,
                           self.dim,
                           device=device,
                           dtype=dtype,
                           generator=self._generator)
        norm = dirs.norm(dim=-1, keepdim=True).clamp_min_(1e-8)
        directions = dirs / norm

        # Cache if we have a generator (for reproducibility)
        if self._generator is not None:
            self._cached_directions = directions.clone()

        return directions


class RolloutRunner:
    """
	Unified rollout runner that can operate in real environment.
	- Environment mode: objective(latents, actions, observations, aux_trace, **kwargs) -> scalar
	"""

    def __init__(
        self,
        planner: Callable[..., Tensor],
        objective: Callable,
        normalizer: LatentNormalizer,
        horizon: int,
        dynamics: Optional[Callable[..., Tuple[Tensor, Any]]] = None,
        env=None,
        encoder: Optional[Callable[..., Tensor]] = None,
        track_gradients: bool = False,
        default_planner_kwargs: Optional[Dict[str, Any]] = None,
        default_dynamics_kwargs: Optional[Dict[str, Any]] = None,
        default_objective_kwargs: Optional[Dict[str, Any]] = None,
        verbose: bool = False,
    ):
        """
		Args:
			planner: Function that maps latent state to action
			objective: Function that computes scalar score from trajectory
				- If env=None: objective(latents, actions, aux_trace, **kwargs) -> scalar
				- If env is set: objective(latents, actions, observations, aux_trace, **kwargs) -> scalar
			normalizer: LatentNormalizer for keeping latents on manifold
			horizon: Rollout length
			dynamics: Function (latent, action) -> (next_latent, aux_info). Required if env=None.
			env: Optional environment for real rollouts. If None, uses world model dynamics.
			encoder: Function to encode observations to latents. Required if env is set.
			renormalize_dynamics: Whether to re-normalize latents after dynamics
			track_gradients: Whether to enable gradient tracking
			default_*_kwargs: Default keyword arguments for respective callables
		"""
        if horizon <= 0:
            raise ValueError("Rollout horizon must be positive.")

        # Mode selection
        self._use_env = env is not None

        if self._use_env:
            if encoder is None:
                raise ValueError(
                    "encoder must be provided when using environment rollouts")
            self._env = env
            self._encoder = encoder
        else:
            if dynamics is None:
                raise ValueError(
                    "dynamics must be provided when not using environment rollouts"
                )
            self._dynamics = dynamics

        self._planner = planner
        self._objective = objective
        self._normalizer = normalizer
        self._horizon = horizon
        self._track_gradients = track_gradients
        self._planner_kwargs = default_planner_kwargs or {}
        self._dynamics_kwargs = default_dynamics_kwargs or {}
        self._objective_kwargs = default_objective_kwargs or {}
        self._verbose = verbose

    def run(
        self,
        z: TensorLike,
        initial_obs: TensorLike,
        initial_state: Dict[str, Any],
        *,
        planner_kwargs: Optional[Dict[str, Any]] = None,
        objective_kwargs: Optional[Dict[str, Any]] = None,
    ) -> RolloutResult:
        """
		Run a rollout from the provided latent state.
		
		Args:
			z: Initial latent state
			initial_obs: Initial observation (only used if env is set)
			planner_kwargs: Additional kwargs for planner
			dynamics_kwargs: Additional kwargs for dynamics (only used if env=None)
			objective_kwargs: Additional kwargs for objective
		"""
        if self._use_env:
            return self._run_environment(
                z=z,
                initial_obs=initial_obs,
                initial_state=initial_state,
                planner_kwargs=planner_kwargs,
                objective_kwargs=objective_kwargs,
            )
        else:
            raise NotImplementedError("Latent rollouts are not supported yet")

    def _run_environment(
        self,
        z: TensorLike,
        initial_obs: Tensor,
        initial_state: Dict[str, Any],
        planner_kwargs: Optional[Dict[str, Any]],
        objective_kwargs: Optional[Dict[str, Any]],
    ) -> RolloutResult:
        """Run rollout in real environment to get true observations."""
        planner_kwargs = {**self._planner_kwargs, **(planner_kwargs or {})}
        objective_kwargs = {
            **self._objective_kwargs,
            **(objective_kwargs or {})
        }

        context = nullcontext(
        ) if self._track_gradients else torch.inference_mode()
        with context:
            # Normalize initial latent
            #z_curr = self._normalizer.normalize(z)
            z_curr = z.clone()

            observations_2 = self._env.reset(initial_state=initial_state,
                                             verbose=self._verbose)
            assert torch.allclose(observations_2, initial_obs)

            latents: List[Tensor] = [z_curr]
            actions: List[Tensor] = []
            observations: List[Tensor] = [initial_obs]
            aux_trace: List[Any] = []

            for _ in range(self._horizon):
                # Plan from current latent
                action = self._planner(z_curr, **planner_kwargs)
                if action.ndim > 1:
                    action = action.squeeze(0)
                actions.append(action)

                # Execute action in environment
                obs_next, reward, done, info = self._env.step(action.cpu())

                # Convert observation to tensor
                if not isinstance(obs_next, torch.Tensor):
                    obs_next = torch.tensor(obs_next,
                                            dtype=torch.float32,
                                            device=z.device)

                observations.append(obs_next)
                aux_trace.append({'reward': reward, 'done': done, 'info': info})

                # Encode next observation to latent
                obs_next_batch = obs_next.unsqueeze(
                    0) if obs_next.ndim == 1 else obs_next
                z_next = self._encoder(obs_next_batch.to(z.device), task=None)
                if z_next.ndim > 1:
                    z_next = z_next.squeeze(0)

                # no normalization, the encoder already simnorms the latents.
                #z_next = self._normalizer.normalize(z_next)

                latents.append(z_next)
                z_curr = z_next

            latents_tensor = torch.stack(latents, dim=0)
            actions_tensor = torch.stack(actions, dim=0)
            observations_tensor = torch.stack(observations, dim=0)

            # Objective receives latents, actions, observations, and aux
            score = self._objective(
                latents_tensor,
                actions_tensor,
                observations_tensor,
                aux_trace,
                **objective_kwargs,
            )

            # Store observations in aux for later access
            trajectory = RolloutTrajectory(latents=latents_tensor,
                                           actions=actions_tensor,
                                           observations=observations_tensor,
                                           aux=aux_trace)
            return RolloutResult(trajectory=trajectory, score=score)


class FiniteDifferenceProbe:
    """
	Random-direction finite difference probe for causal latent gradients.
	
	If a seed is provided, the probe will produce identical results across
	multiple calls to estimate() with the same inputs (deterministic planning).
	"""

    def __init__(
        self,
        rollout_runner: RolloutRunner,
        normalizer: LatentNormalizer,
        num_directions: int,
        perturbation_scale: float,
        direction_sampler: Optional[RandomDirectionSampler] = None,
        seed: Optional[int] = None,
    ):
        if perturbation_scale <= 0:
            raise ValueError("perturbation_scale must be positive.")
        self._runner = rollout_runner
        self._normalizer = normalizer
        self._num_directions = num_directions
        self._epsilon = perturbation_scale
        self._direction_sampler = direction_sampler
        self._seed = seed

    def estimate(self,
                 z: TensorLike,
                 initial_obs: TensorLike,
                 initial_state: Dict[str, Any],
                 directions: Optional[Tensor] = None,
                 seed: Optional[int] = None) -> CausalGradientResult:
        """
		Estimate dJ/dz around the provided latent state using SPSA-style finite differences.
		
		Args:
			z: Initial latent state
			initial_obs: Initial observation
			initial_state: Initial state dict for environment reset
			directions: Optional pre-computed direction vectors. If None, will sample new directions.
			seed: Optional seed for this specific estimate call. If provided, overrides the probe's default seed.
		
		Note:
			The seed parameter only affects the planner (MPPI sampling, policy prior, action selection).
			Direction sampling is independent and controlled by the direction_sampler's generator.
			If no generator was provided to the direction_sampler, directions will be sampled randomly
			each time, unaffected by the seed parameter.
		"""

        device = z.device
        dtype = z.dtype

        # Use provided seed, fall back to probe's default seed
        active_seed = seed if seed is not None else self._seed

        # Set seed for planner reproducibility
        if active_seed is not None:
            set_seed(active_seed)

        if directions is None:
            # Sample directions using current random state
            directions = self._direction_sampler.sample(device=device,
                                                        dtype=dtype)
        else:
            # Validate provided directions
            if directions.shape != (self._num_directions, z.shape[-1]):
                raise ValueError(
                    f"Provided directions shape {directions.shape} doesn't match "
                    f"expected shape ({self._num_directions}, {z.shape[-1]})")

        baseline = self._runner.run(z,
                                    initial_obs=initial_obs,
                                    initial_state=initial_state)
        gradient = torch.zeros_like(z)
        positive_results: List[RolloutResult] = []
        negative_results: List[RolloutResult] = []

        # NOTE(R): Is there a way to batch this?
        # I dont think so because dm environemnt dont' support vectorization.
        for direction in directions:
            perturbation = self._epsilon * direction
            z_pos = self._normalizer.project_no_norm(z, perturbation)
            z_neg = self._normalizer.project_no_norm(z, -perturbation)

            result_pos = self._runner.run(z_pos,
                                          initial_obs=initial_obs,
                                          initial_state=initial_state)
            result_neg = self._runner.run(z_neg,
                                          initial_obs=initial_obs,
                                          initial_state=initial_state)
            positive_results.append(result_pos)
            negative_results.append(result_neg)

            score_delta = (result_pos.score -
                           result_neg.score) / (2 * self._epsilon)
            gradient = gradient + score_delta * direction

        gradient = gradient / directions.shape[0]
        return CausalGradientResult(
            gradient=gradient,
            baseline=baseline,
            directions=directions,
            perturbation_scale=self._epsilon,
            positive_rollouts=positive_results,
            negative_rollouts=negative_results,
        )


class LatentObjectiveRunner:
    """
	Runner for computing objectives directly from latent states without rollouts.
	
	This is useful when:
	1. The objective is a direct function of the latent (e.g., latent[0] for cart position)
	2. You want to decode the latent to get specific features, then compute objective
	3. You want to backpropagate through the decoder
	
	The objective can be:
	- A function of latents: objective(z, **kwargs) -> scalar
	- A function of decoded features: objective(decoded_features, **kwargs) -> scalar
	"""

    def __init__(
        self,
        objective: Callable,
        decoder: Optional[Callable[[Tensor], Tensor]] = None,
        track_gradients: bool = True,
        default_objective_kwargs: Optional[Dict[str, Any]] = None,
    ):
        """
		Args:
			objective: Function that computes scalar score
				- If decoder=None: objective(z, **kwargs) -> scalar
				- If decoder is set: objective(decoded_features, **kwargs) -> scalar
			decoder: Optional function to decode latent to features
			track_gradients: Whether to enable gradient tracking (needed for backprop)
			default_objective_kwargs: Default keyword arguments for objective
		"""
        self._objective = objective
        self._decoder = decoder
        self._track_gradients = track_gradients
        self._objective_kwargs = default_objective_kwargs or {}

    def run(
        self,
        z: TensorLike,
        *,
        objective_kwargs: Optional[Dict[str, Any]] = None,
    ) -> LatentObjectiveResult:
        """
		Evaluate the objective on the latent state.
		
		Args:
			z: Latent state
			objective_kwargs: Additional kwargs for objective
			
		Returns:
			LatentObjectiveResult containing latent, decoded features (if decoder), and score
		"""
        objective_kwargs = {
            **self._objective_kwargs,
            **(objective_kwargs or {})
        }

        context = nullcontext(
        ) if self._track_gradients else torch.inference_mode()

        with context:
            if self._decoder is not None:
                # Decode latent and compute objective on decoded features
                decoded = self._decoder(z)
                score = self._objective(decoded, **objective_kwargs)
            else:
                # Compute objective directly on latent
                decoded = None
                score = self._objective(z, **objective_kwargs)

            return LatentObjectiveResult(latent=z, decoded=decoded, score=score)


class BackpropGradientProbe:
    """
	Gradient probe that uses backpropagation to compute exact gradients dJ/dz.
	
	This is more efficient and accurate than finite differences when:
	1. The objective is differentiable
	2. You have a differentiable decoder
	3. You want exact gradients rather than SPSA estimates
	
	Usage:
		probe = BackpropGradientProbe(
			objective_runner=latent_objective_runner,
		)
		result = probe.estimate(z)  # Computes gradient via backprop
	"""

    def __init__(
        self,
        objective_runner: LatentObjectiveRunner,
    ):
        """
		Args:
			objective_runner: LatentObjectiveRunner for computing objective
		"""
        self._runner = objective_runner

    def estimate(
        self,
        z: TensorLike,
        *,
        objective_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Tensor, LatentObjectiveResult]:
        """
		Compute dJ/dz using backpropagation.
		
		Args:
			z: Latent state (must have requires_grad=True to compute gradients)
			objective_kwargs: Additional kwargs for objective
			
		Returns:
			Tuple of (gradient, result)
			- gradient: dJ/dz computed via backprop
			- result: LatentObjectiveResult with score and decoded features
		"""
        # Ensure z requires gradients
        if not z.requires_grad:
            z = z.detach().clone().requires_grad_(True)

        # Run objective
        result = self._runner.run(z, objective_kwargs=objective_kwargs)

        # Backpropagate to get gradient
        result.score.backward()

        # Extract gradient
        gradient = z.grad.clone() if z.grad is not None else torch.zeros_like(z)

        return gradient, result
