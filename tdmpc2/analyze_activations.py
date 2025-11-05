"""
Script for recording episode and planning data during TD-MPC2 agent execution.
Used for mechanistic interpretability analysis.

Records:
- Episode data: observations, actions, rewards, latent states
- Planning data: MPPI iterations, elite trajectories, action distributions (optional)

Example usage:
	$ python analyze_activations.py task=dog-run checkpoint=path/to/checkpoint.pt num_episodes=10
	$ python analyze_activations.py task=dog-run checkpoint=path/to/checkpoint.pt num_episodes=5 record_planning=true
"""

import os

os.environ['MUJOCO_GL'] = os.getenv("MUJOCO_GL", 'egl')
os.environ['LAZY_LEGACY_OP'] = '0'
import warnings

warnings.filterwarnings('ignore')

import torch
import hydra
from pathlib import Path
from termcolor import colored
from tqdm import tqdm

from common.parser import parse_cfg
from common.seed import set_seed
from common.episode_data_recorder import EpisodeDataRecorder
from common.planning_data_recorder import PlanningDataRecorder
from envs import make_env
from recording_tdmpc2 import RecordingTDMPC2


@hydra.main(config_name='config', config_path='.')
def analyze_activations(cfg: dict):
    """
	Record episode and planning data during agent execution for interpretability analysis.
	
	Args (via config):
		task: Task name
		checkpoint: Path to model checkpoint
		num_episodes: Number of episodes to record (default: 10)
		record_planning: Whether to record MPPI planning data (default: false)
		save_dir: Directory to save recordings (default: analysis/activations/{task})
		seed: Random seed
		model_size: Model size (for multi-task models)
	
	Example:
		$ python analyze_activations.py task=dog-run checkpoint=ckpts/dog-1.pt num_episodes=10
		$ python analyze_activations.py task=dog-run checkpoint=ckpts/dog-1.pt record_planning=true
	"""

    # Parse config
    cfg = parse_cfg(cfg)
    set_seed(cfg.seed)

    # Set defaults for analysis
    num_episodes = cfg.get('num_episodes', 1)
    record_planning = cfg.get('record_planning', False)
    base_save_dir = cfg.get('save_dir', f'analysis/{cfg.task}')

    print(colored('=' * 80, 'cyan'))
    print(colored('TD-MPC2 Data Recording', 'cyan', attrs=['bold']))
    print(colored('=' * 80, 'cyan'))
    print(f"Task: {cfg.task}")
    print(f"Checkpoint: {cfg.checkpoint}")
    print(f"Num episodes: {num_episodes}")
    print(f"Record planning: {record_planning}")
    print(f"Save directory: {base_save_dir}")
    print(colored('=' * 80, 'cyan'))

    # Initialize environment
    print("\nInitializing environment...")
    env = make_env(cfg)

    # Initialize recording agent (recorders will be set per episode)
    print("\nInitializing recording agent...")
    agent = RecordingTDMPC2(cfg, episode_recorder=None, planning_recorder=None)

    # Load checkpoint
    if cfg.checkpoint and cfg.checkpoint != '???':
        print(f"Loading checkpoint from {cfg.checkpoint}...")
        agent.load(cfg.checkpoint)
    else:
        raise ValueError(
            "Must provide a checkpoint path via checkpoint=path/to/checkpoint.pt"
        )

    # Run episodes and record activations
    print(f"\nRecording activations for {num_episodes} episodes...")
    print(colored('-' * 80, 'yellow'))

    all_episode_rewards = []
    all_episode_lengths = []
    all_episode_successes = []

    for ep_idx in tqdm(range(num_episodes), desc="Recording episodes"):
        # Create episode-specific save directories
        episode_dir = Path(base_save_dir) / f'episode_{ep_idx}_seed_{cfg.seed}'
        activations_dir = episode_dir / 'activations'
        planning_dir = episode_dir / 'planning'

        # Create recorders for this episode
        episode_recorder = EpisodeDataRecorder(cfg,
                                               save_dir=str(activations_dir))

        if record_planning:
            planning_recorder = PlanningDataRecorder(
                save_full_trajectories=False, save_dir=str(planning_dir))
        else:
            planning_recorder = None

        # Update agent's recorders for this episode
        agent.episode_recorder = episode_recorder
        agent.planning_recorder = planning_recorder

        # Reset environment
        obs = env.reset()
        task_arg = None

        done = False
        ep_reward = 0
        t = 0

        # Run episode
        while not done:
            # Act (agent records planning data internally if enabled)
            action = agent.act(obs, t0=(t == 0), eval_mode=True, task=task_arg)

            # Step environment
            next_obs, reward, done, info = env.step(action)

            # Get latent state from agent for episode recording
            latent_state = agent.get_last_latent()

            # Record episode data
            episode_recorder.record_step(obs=obs,
                                         action=action,
                                         reward=reward,
                                         done=done,
                                         latent_state=latent_state)

            # Update for next step
            obs = next_obs
            ep_reward += reward
            t += 1

        # Save episode data
        metadata = {
            'episode_idx': ep_idx,
            'episode_reward': float(ep_reward),
            'episode_length': t,
            'success': bool(info.get('success', False)),
            'checkpoint': str(cfg.checkpoint),
            'seed': cfg.seed,
        }

        episode_recorder.save_episode(metadata=metadata)

        # Save planning data if recording
        if planning_recorder is not None:
            planning_recorder.save_episode(metadata=metadata)

        # Track statistics
        all_episode_rewards.append(ep_reward)
        all_episode_lengths.append(t)
        all_episode_successes.append(info.get('success', False))

    # Print summary statistics
    print(colored('-' * 80, 'yellow'))
    print(colored('\nRecording Summary:', 'green', attrs=['bold']))
    print(f"Episodes recorded: {num_episodes}")
    print(
        f"Average reward: {sum(all_episode_rewards)/len(all_episode_rewards):.2f} ± {torch.tensor(all_episode_rewards).std():.2f}"
    )
    print(
        f"Average length: {sum(all_episode_lengths)/len(all_episode_lengths):.1f}"
    )
    print(
        f"Success rate: {sum(all_episode_successes)/len(all_episode_successes)*100:.1f}%"
    )
    print(f"\nData saved to: {base_save_dir}")
    print(colored('=' * 80, 'cyan'))

    # Save summary statistics
    summary = {
        'num_episodes': num_episodes,
        'task': cfg.task,
        'checkpoint': str(cfg.checkpoint),
        'seed': cfg.seed,
        'record_planning': record_planning,
        'statistics': {
            'mean_reward':
                float(sum(all_episode_rewards) / len(all_episode_rewards)),
            'std_reward':
                float(torch.tensor(all_episode_rewards).std()),
            'mean_length':
                float(sum(all_episode_lengths) / len(all_episode_lengths)),
            'success_rate':
                float(sum(all_episode_successes) / len(all_episode_successes)),
        },
        'episode_rewards': [float(r) for r in all_episode_rewards],
        'episode_lengths': [int(l) for l in all_episode_lengths],
        'episode_successes': [bool(s) for s in all_episode_successes],
    }

    import json
    summary_file = Path(base_save_dir) / 'summary.json'
    summary_file.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"Summary saved to: {summary_file}")


if __name__ == '__main__':
    analyze_activations()
