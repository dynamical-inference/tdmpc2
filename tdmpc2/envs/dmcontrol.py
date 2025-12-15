from collections import defaultdict, deque

import gymnasium as gym
import numpy as np
import torch

from envs.tasks import cheetah, walker, hopper, reacher, ball_in_cup, pendulum, fish, cartpole
from dm_control import suite

suite.ALL_TASKS = suite.ALL_TASKS + suite._get_tasks('custom')
suite.TASKS_BY_DOMAIN = suite._get_tasks_by_domain(suite.ALL_TASKS)
from dm_control.suite.wrappers import action_scale

from envs.wrappers.timeout import Timeout


def get_obs_shape(env):
    obs_shp = []
    for v in env.observation_spec().values():
        try:
            shp = np.prod(v.shape)
        except:
            shp = 1
        obs_shp.append(shp)
    return (int(np.sum(obs_shp)),)


class DMControlWrapper:

    def __init__(self, env, domain, initial_state=None):
        """
        Initialize DMControl wrapper with optional initial state.
        
        Args:
            env: DM Control environment
            domain: Domain name (e.g., 'cheetah', 'walker')
            initial_state: Dict with state variables to set on reset.
                Currently supported:
                - 'xpos': Cartesian positions of bodies (dict mapping body_name -> [x,y,z])
                
                Easy to extend to: qpos, qvel, qacc, act, ctrl, mocap_pos, etc.
                
                Example:
                    initial_state = {
                        'xpos': {'torso': [0.0, 0.0, 1.0]}  # Set torso position
                    }
        """
        self.env = env
        self.camera_id = 2 if domain == 'quadruped' else 0
        obs_shape = get_obs_shape(env)
        action_shape = env.action_spec().shape
        self.observation_space = gym.spaces.Box(low=np.full(obs_shape,
                                                            -np.inf,
                                                            dtype=np.float32),
                                                high=np.full(obs_shape,
                                                             np.inf,
                                                             dtype=np.float32),
                                                dtype=np.float32)
        self.action_space = gym.spaces.Box(
            low=np.full(action_shape,
                        env.action_spec().minimum),
            high=np.full(action_shape,
                         env.action_spec().maximum),
            dtype=env.action_spec().dtype)
        self.action_spec_dtype = env.action_spec().dtype
        # Store initial state for custom resets
        self._initial_state = initial_state if initial_state is not None else {}

    @property
    def unwrapped(self):
        return self.env

    def _obs_to_array(self, obs):
        return torch.from_numpy(
            np.concatenate([v.flatten() for v in obs.values()],
                           dtype=np.float32))

    def reset(self, initial_state=None, verbose=False):
        """Reset environment and optionally apply custom initial state."""
        obs = self.env.reset().observation

        # Apply custom initial state if provided
        initial_state = initial_state if initial_state is not None else self._initial_state
        if initial_state:
            assert isinstance(initial_state,
                              dict), "initial_state must be a dict"
            physics = self.env.physics
            modified = False

            if 'qpos' in initial_state:
                modified = True
                self._set_joint_positions(physics, initial_state['qpos'])
                if verbose:
                    print('Intial position modified!')
            if 'qvel' in initial_state:
                modified = True
                self._set_joint_velocities(physics, initial_state['qvel'])
                if verbose:
                    print("Initial velocities modified!")

            # Update physics and observation if we modified anything
            if modified:
                physics.forward()
                obs = self.env.task.get_observation(physics)
        return self._obs_to_array(obs)

    def _set_joint_positions(self, physics, qpos_dict):
        for name, val in qpos_dict.items():
            physics.named.data.qpos[name] = val

    def _set_joint_velocities(self, physics, qvel_dict):
        for name, val in qvel_dict.items():
            physics.named.data.qvel[name] = val

    def step(self, action):
        reward = 0
        action = action.astype(self.action_spec_dtype)
        for _ in range(2):
            step = self.env.step(action)
            reward += step.reward
        return self._obs_to_array(
            step.observation), reward, False, defaultdict(float)

    def render(self, width=384, height=384, camera_id=None):
        return self.env.physics.render(height, width, camera_id or
                                       self.camera_id)


class Pixels(gym.Wrapper):

    def __init__(self, env, cfg, num_frames=3, size=64):
        super().__init__(env)
        self.cfg = cfg
        self.env = env
        self.observation_space = gym.spaces.Box(low=0,
                                                high=255,
                                                shape=(num_frames * 3, size,
                                                       size),
                                                dtype=np.uint8)
        self._frames = deque([], maxlen=num_frames)
        self._size = size

    def _get_obs(self, is_reset=False):
        frame = self.env.render(width=self._size,
                                height=self._size).transpose(2, 0, 1)
        num_frames = self._frames.maxlen if is_reset else 1
        for _ in range(num_frames):
            self._frames.append(frame)
        return torch.from_numpy(np.concatenate(self._frames))

    def reset(self):
        self.env.reset()
        return self._get_obs(is_reset=True)

    def step(self, action):
        _, reward, done, info = self.env.step(action)
        return self._get_obs(), reward, done, info


def make_env(cfg):
    """
    Make DMControl environment.
    Adapted from https://github.com/facebookresearch/drqv2
    """
    domain, task = cfg.task.replace('-', '_').split('_', 1)
    domain = dict(cup='ball_in_cup', pointmass='point_mass').get(domain, domain)
    if (domain, task) not in suite.ALL_TASKS:
        raise ValueError('Unknown task:', task)
    assert cfg.obs in {'state', 'rgb'
                      }, 'This task only supports state and rgb observations.'
    env = suite.load(domain,
                     task,
                     task_kwargs={'random': cfg.seed},
                     visualize_reward=False)
    env = action_scale.Wrapper(env, minimum=-1., maximum=1.)
    # Get initial state from config if provided
    initial_state = getattr(cfg, 'initial_state', None)
    env = DMControlWrapper(env, domain, initial_state=initial_state)
    if cfg.obs == 'rgb':
        env = Pixels(env, cfg)
    env = Timeout(env, max_episode_steps=500)
    return env
