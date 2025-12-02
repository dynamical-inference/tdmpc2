import os

os.environ["MUJOCO_GL"] = "egl"  # Must be set BEFORE importing dm_control

from rastermap import Rastermap
import matplotlib.pyplot as plt
import numpy as np
from dm_control import suite
from tqdm import tqdm
import imageio


def extract_variables(data,
                      variables=['latent_states', 'observations'],
                      n_episodes=None,
                      n_steps=None):
    out = {var: [] for var in variables}
    episodes = data.get('episodes', [])

    if n_episodes is not None:
        episodes = episodes[:n_episodes]

    for ep in episodes:
        d = ep['data']
        for var in variables:
            arr = d.get(var)
            arr = np.array(arr)[:n_steps]
            if var == 'latent_states' and arr.ndim == 3 and arr.shape[1] == 1:
                arr = arr.squeeze(1)
            if var == 'observations' and arr.ndim == 1:
                arr = arr.reshape(-1, 1)
            if var == 'actions' and arr.ndim == 1:
                arr = arr.reshape(-1, 1)
            if var == 'rewards' and arr.ndim == 1:
                arr = arr.reshape(-1, 1)
            out[var].append(arr)
    return out


def fit_rastermap(data,
                  n_PCs=200,
                  n_clusters=100,
                  locality=0.75,
                  time_lag_window=10):
    """Fit Rastermap model to data and return embedding results."""
    model = Rastermap(n_PCs=n_PCs,
                      n_clusters=n_clusters,
                      locality=locality,
                      time_lag_window=time_lag_window).fit(data)

    y = model.embedding  # neurons x 1
    isort = model.isort
    # visualize binning over neurons
    X_embedding = model.X_embedding

    return model, y, isort, X_embedding


def plot_rastermap_with_rewards_actions(X_embedding=None,
                                        isort=None,
                                        rewards=None,
                                        actions=None,
                                        latents=None,
                                        plot_type='embedding',
                                        points_to_plot=None,
                                        sort_latents=True,
                                        vmin=0,
                                        vmax=1):

    # Apply points_to_plot filtering first
    if points_to_plot is not None:
        rewards_to_plot = rewards[:points_to_plot]
        actions_to_plot = actions[:points_to_plot]
        if latents is not None:
            latents = latents[:, :points_to_plot]
        if X_embedding is not None:
            X_embedding = X_embedding[:, :points_to_plot]

    else:
        rewards_to_plot = rewards
        actions_to_plot = actions

    # Create figure with subplots
    fig = plt.figure(figsize=(14, 10), dpi=150)

    # Create GridSpec for subplots (10 rows for rewards, actions, and main plot)
    grid = plt.GridSpec(12, 1, figure=fig, hspace=0.3)

    # Rewards subplot
    ax_rewards = plt.subplot(grid[0:1, 0])
    ax_rewards.plot(rewards_to_plot, color='blue')
    ax_rewards.set_xlim([0, len(rewards_to_plot)])
    ax_rewards.set_ylabel("rewards", fontsize=8)
    #ax_rewards.set_title(f"Episode {episode_idx}", color='blue', fontsize=10)
    ax_rewards.axis("off")

    # Actions subplot
    ax_actions = plt.subplot(grid[1:2, 0])
    ax_actions.plot(actions_to_plot, color='gray')
    ax_actions.set_xlim([0, len(actions_to_plot)])
    ax_actions.set_ylabel("actions", fontsize=8)
    ax_actions.axis("off")

    # Main plot (embedding/latents/superneuron)
    ax_main = plt.subplot(grid[2:10, 0])

    if plot_type == 'embedding':
        im = ax_main.imshow(X_embedding,
                            vmin=0,
                            vmax=1.5,
                            cmap="gray_r",
                            aspect="auto",
                            interpolation='none')
        ax_main.set_xlabel("Time", fontsize=8)
        ax_main.set_ylabel("Embedding bins", fontsize=8)

    elif plot_type == 'latents':
        if not sort_latents:
            latents_rm_sorted = latents
        else:
            latents_rm_sorted = latents[isort]
        im = ax_main.imshow(latents_rm_sorted,
                            vmin=vmin,
                            vmax=vmax,
                            cmap="gray_r",
                            aspect="auto",
                            interpolation='none')
        ax_main.set_xlabel("Time", fontsize=8)
        ax_main.set_ylabel("Latent dimension", fontsize=8)

    elif plot_type == 'superneuron':
        if not sort_latents:
            latents_rm_sorted = latents
        else:
            latents_rm_sorted = latents[isort]
        nbin = 8  # number of neurons to bin over
        ndiv = (latents_rm_sorted.shape[0] // nbin) * nbin
        # group sorted matrix into rows of length nbin
        sn = latents_rm_sorted[:ndiv].reshape(ndiv // nbin, nbin, -1)
        # take mean over neurons in a bin
        sn = sn.mean(axis=1)

        im = ax_main.imshow(sn,
                            vmin=0,
                            vmax=1,
                            cmap="gray_r",
                            aspect="auto",
                            interpolation='none')
        ax_main.set_xlabel("Time", fontsize=8)
        ax_main.set_ylabel(f"Superneuron bins (n={nbin})", fontsize=8)

    # Add colorbar
    plt.colorbar(im, ax=ax_main, fraction=0.046, pad=0.04)

    return fig


def overlap_frames(frames, pole_alpha=0.3, base_alpha=0.3):
    """
    Overlay multiple frames to show pole trajectory.
    
    Args:
        frames: List of frames to overlay
        pole_alpha: Visibility of the pole overlay (0=invisible, 1=fully visible)
        base_alpha: Visibility of the first frame (0=invisible, 1=fully visible)
    """
    # Filter out None values
    valid_frames = [f for f in frames if f is not None]

    if not valid_frames:
        raise ValueError("No valid frames to overlap")

    # Convert frames to float for processing
    frames_float = [f.astype(float) for f in valid_frames]

    # Use the first frame as the base with reduced visibility
    base_frame = frames_float[0].copy() * base_alpha

    # For subsequent frames, blend them in with reduced opacity
    for frame in frames_float[1:]:
        # Compute the maximum to capture the pole positions
        pole_overlay = np.maximum(base_frame, frame * pole_alpha)
        # Blend: keep more of the base, add pole with reduced alpha
        base_frame = base_frame * (1 - pole_alpha) + pole_overlay * pole_alpha

    # Convert back to uint8
    overlapped = np.clip(base_frame, 0, 255).astype('uint8')

    # Display the overlapped image
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    ax.imshow(overlapped)
    ax.axis('off')
    plt.tight_layout()
    plt.show()

    return overlapped


def compute_success(df_data, threshold_angle=0.1, consecutive_steps=100):
    """
    Compute whether the pole is upright (straight) for consecutive_steps time points.
    
    Args:
        df_data: DataFrame with 'observations' column containing state arrays
        threshold_angle: Maximum angle deviation from vertical (in radians) to consider "straight"
        consecutive_steps: Number of consecutive steps required for success
    
    Returns:
        df_data: DataFrame with added 'success' column
    """
    success = []

    for idx in range(len(df_data)):
        observations = np.array(df_data['observations'].iloc[idx])

        # Extract cos and sin of pole angle (dimensions 1 and 2)
        cos_angle = observations[:, 1]
        sin_angle = observations[:, 2]

        # Compute actual angle from cos and sin
        angles = np.arctan2(sin_angle, cos_angle)

        # Check if angle is within threshold (close to 0, which is upright)
        is_straight = np.abs(angles) < threshold_angle

        # Check for consecutive_steps consecutive True values
        episode_success = False
        count = 0
        for straight in is_straight:
            if straight:
                count += 1
                if count >= consecutive_steps:
                    episode_success = True
                    break
            else:
                count = 0

        success.append(episode_success)

    df_data['success'] = np.array(success)
    return df_data


def reconstruct_frame(states, env, height=240, width=320, camera_id=0):
    env.physics.data.qpos[0] = states[0]
    env.physics.data.qpos[1] = states[1]
    env.physics.data.qvel[0] = states[2]
    env.physics.data.qvel[1] = states[3]
    env.physics.forward()
    frame = env.physics.render(height=height, width=width, camera_id=camera_id)
    return frame


def reconstruct_cartpole_dmcontrol(states,
                                   height=240,
                                   width=320,
                                   camera_id=0,
                                   save_video=False,
                                   out_path=None):

    if save_video:
        assert out_path is not None, "out_path must be provided if save_video is True"

    env = suite.load("cartpole", "swingup")
    pole_angle_radians = np.arctan2(states[:, 2], states[:, 1]).reshape(-1, 1)
    states_radians = np.concatenate([
        states[:, 0].reshape(-1, 1),
        pole_angle_radians.reshape(-1, 1), states[:, 2:4]
    ],
                                    axis=1)
    assert states_radians.shape[1] == 4

    frames = []
    for state in tqdm(states_radians):
        frame = reconstruct_frame(state, env, height, width, camera_id)
        frames.append(frame)

    if save_video:
        assert out_path is not None, "out_path must be provided if save_video is True"
        imageio.mimwrite(out_path, frames, fps=30, macro_block_size=None)
        print(f"Wrote {len(frames)} frames to {os.path.abspath(out_path)}")
    return frames
