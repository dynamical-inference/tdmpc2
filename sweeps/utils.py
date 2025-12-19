from envs import make_env
from recording_tdmpc2 import RecordingTDMPC2
import imageio
import os


def run_episode_with_recording(env,
                               agent,
                               episode_recorder=None,
                               planning_recorder=None,
                               save_video=False,
                               eval_mode=True,
                               task=None,
                               episode_dir=None,
                               task_name=None,
                               patcher=None):
    """
    Run a single episode with optional recording.
    
    This function encapsulates the episode execution logic and can be used by
    both baseline recordings and intervention experiments.
    
    Args:
        env: Environment instance
        agent: RecordingTDMPC2 agent
        episode_recorder: Optional EpisodeDataRecorder instance
        planning_recorder: Optional PlanningDataRecorder instance
        save_video: Whether to capture video frames
        eval_mode: Whether to run in evaluation mode
        task: Task index (for multi-task models)
        patcher: Optional ActivationPatcher instance (for timestep tracking)
        
    Returns:
        episode_stats: Dict with reward, length, success, frames
    """
    # Update agent's recorders
    agent.planning_recorder = planning_recorder

    # Reset environment to the specified task
    if task is not None:
        # NOTE(R): This is necessary for multi-task environments
        obs = env.reset(task_idx=task)
    else:
        obs = env.reset()
    done = False
    ep_reward = 0
    t = 0
    frames = []

    # Reset patcher timestep if provided
    if patcher is not None:
        patcher.reset_step()

    if save_video:
        frames.append(env.render())

    # Run episode
    while not done:
        # Act (agent records planning data internally if enabled)
        action = agent.act(obs, t0=(t == 0), eval_mode=eval_mode, task=task)

        # Step environment
        next_obs, reward, done, info = env.step(action)

        # Record episode data if recorder is attached
        if episode_recorder is not None:
            latent_state = agent.get_last_latent()
            episode_recorder.record_step(obs=obs,
                                         action=action,
                                         reward=reward,
                                         done=done,
                                         latent_state=latent_state)

        # Update for next step
        obs = next_obs
        ep_reward += reward
        t += 1

        # Increment patcher timestep if provided
        if patcher is not None:
            patcher.increment_step()

        if save_video:
            frames.append(env.render())

    if save_video:
        task_name = task_name if task_name is not None else 'task_video'
        imageio.mimsave(os.path.join(episode_dir, f'{task_name}.mp4'),
                        frames,
                        fps=30)


def setup_agent(cfg):
    """
    Initialize environment and agent.
    
    Args:
        cfg: Configuration object
        
    Returns:
        env: Environment instance
        agent: RecordingTDMPC2 agent instance
    """
    # Initialize environment
    env = make_env(cfg)

    # Initialize recording agent
    agent = RecordingTDMPC2(cfg, planning_recorder=None)

    # Load checkpoint
    if cfg.checkpoint and cfg.checkpoint != '???':
        agent.load(cfg.checkpoint)
    else:
        raise ValueError(
            "Must provide a checkpoint path via checkpoint=path/to/checkpoint.pt"
        )

    return env, agent
