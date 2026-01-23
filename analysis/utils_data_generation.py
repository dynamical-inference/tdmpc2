import numpy as np
import torch
from tqdm import tqdm


def wrap_pi(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi


def state_to_x_th(obs_vec):
    """
    obs_vec: [x, cos(th), sin(th), dx, dth]
    returns x, dx, th, dth
    """
    x = float(obs_vec[0])
    cth = float(obs_vec[1])
    sth = float(obs_vec[2])
    dx = float(obs_vec[3])
    dth = float(obs_vec[4])
    th = np.arctan2(sth, cth)
    th = wrap_pi(th)
    return x, dx, th, dth


class OUNoise:

    def __init__(self, theta=0.15, sigma=0.4, mu=0.0, dt=1.0, rng=None):
        self.theta = float(theta)
        self.sigma = float(sigma)
        self.mu = float(mu)
        self.dt = float(dt)
        self.rng = np.random.default_rng() if rng is None else rng
        self.x = 0.0

    def reset(self):
        self.x = 0.0

    def sample(self):
        self.x += self.theta * (self.mu - self.x) * self.dt
        self.x += self.sigma * np.sqrt(self.dt) * float(
            self.rng.standard_normal())
        return self.x


def energy_shaping_u(obs_vec, kE=1.5, kx=0.15, kdx=0.05):
    """
    Heuristic swing-up: pump energy using th, dth; keep cart bounded via x, dx terms.
    """
    x, dx, th, dth = state_to_x_th(obs_vec)

    # energy proxy (dimensionless-ish): 0.5*dth^2 + (1 - cos(th))
    E = 0.5 * (dth**2) + (1.0 - np.cos(th))
    E_des = 2.0

    pump_dir = np.sign(dth * np.cos(th) + 1e-6)
    u = kE * (E_des - E) * pump_dir
    u += -kx * x - kdx * dx
    return float(u)


def pd_upright_u(obs_vec, kp_th=5.0, kd_th=0.8, kp_x=0.2, kd_x=0.1):
    """
    Simple PD around upright th~0 plus small cart centering.
    """
    x, dx, th, dth = state_to_x_th(obs_vec)
    u = -kp_th * th - kd_th * dth - kp_x * x - kd_x * dx
    return float(u)


class DMatchPolicy:
    """
    Episode-level mixture:
      A) OU noise
      B) Kick then OU
      C) Swing-up -> stabilize -> small OU (re-swing if not upright)
    obs_vec must be [x, cos(th), sin(th), dx, dth].
    """

    def __init__(
            self,
            action_low=-1.0,
            action_high=1.0,
            p_mode=(0.50, 0.5),
            ou_theta=0.15,
            ou_sigma_A=0.40,
            ou_sigma_B=0.25,
            ou_sigma_C=0.15,
            kick_steps=(10, 20),
            kick_flip_prob=0.25,
            swingup_steps=(100, 200),
            stabilize_steps=(50, 100),
            upright_theta_thresh=0.35,  # rad
            upright_dtheta_thresh=2.0,  # rad/s
            action_smooth=0.90,  # low-pass; closer to 1 => smoother
            rng=None):
        self.low = float(action_low)
        self.high = float(action_high)
        self.rng = np.random.default_rng() if rng is None else rng

        p = np.asarray(p_mode, dtype=np.float64)
        self.p_mode = p / p.sum()

        self.ou_A = OUNoise(theta=ou_theta, sigma=ou_sigma_A, rng=self.rng)
        self.ou_B = OUNoise(theta=ou_theta, sigma=ou_sigma_B, rng=self.rng)
        self.ou_C = OUNoise(theta=ou_theta, sigma=ou_sigma_C, rng=self.rng)

        self.kick_steps = kick_steps
        self.kick_flip_prob = float(kick_flip_prob)
        self.swingup_steps = swingup_steps
        self.stabilize_steps = stabilize_steps
        self.upright_theta_thresh = float(upright_theta_thresh)
        self.upright_dtheta_thresh = float(upright_dtheta_thresh)

        self.alpha = float(action_smooth)
        self.a_prev = 0.0

        self.mode = None
        self.t = 0
        self.kick_len = 0
        self.kick_sign = 1.0
        self.kick_flip_t = None
        self.swing_len = 0
        self.stab_len = 0

    def reset_episode(self, mode=None):
        self.t = 0
        if mode is None:
            self.mode = int(self.rng.choice([0, 1], p=self.p_mode))
        else:
            self.mode = mode
        self.a_prev = 0.0
        self.ou_A.reset()
        self.ou_B.reset()
        self.ou_C.reset()

        if self.mode == 1:
            self.kick_len = int(
                self.rng.integers(self.kick_steps[0], self.kick_steps[1] + 1))
            self.kick_sign = float(self.rng.choice([-1.0, 1.0]))
            self.kick_flip_t = int(self.rng.integers(1, self.kick_len)) if (
                self.rng.random() < self.kick_flip_prob) else None

        if self.mode == 2:
            self.swing_len = int(
                self.rng.integers(self.swingup_steps[0],
                                  self.swingup_steps[1] + 1))
            self.stab_len = int(
                self.rng.integers(self.stabilize_steps[0],
                                  self.stabilize_steps[1] + 1))

    def act(self, obs_vec):
        if self.mode == 0:
            a = self.ou_A.sample()

        elif self.mode == 1:
            if self.t < self.kick_len:
                sgn = self.kick_sign
                if self.kick_flip_t is not None and self.t >= self.kick_flip_t:
                    sgn = -sgn
                a = sgn
            else:
                a = self.ou_B.sample()

        else:
            # Swing-up -> stabilize -> small OU; re-swing if drifted away
            if self.t < self.swing_len:
                a = energy_shaping_u(obs_vec)
            elif self.t < self.swing_len + self.stab_len:
                a = pd_upright_u(obs_vec)
            else:
                x, dx, th, dth = state_to_x_th(obs_vec)
                if (abs(th) > self.upright_theta_thresh) or (
                        abs(dth) > self.upright_dtheta_thresh):
                    a = energy_shaping_u(obs_vec)
                else:
                    a = self.ou_C.sample()

        # --- ADD CART CENTERING HERE ---
        x, dx, _, _ = state_to_x_th(obs_vec)
        kx, kdx = 0.3, 0.25  # start here; tune
        a = float(a) - kx * x - kdx * dx

        # Low-pass filter + clip
        a = self.alpha * self.a_prev + (1.0 - self.alpha) * float(a)
        self.a_prev = a
        a = float(np.clip(a, self.low, self.high))

        self.t += 1
        return np.array([a], dtype=np.float32)


def collect_data(
    env,
    initial_states,
    n_episodes_per_initial_state,
    max_steps=500,
    seed=0,
    is_rgb=False,
):

    episodes = []  # list of episodes; each episode is list of transitions
    for initial_state in tqdm(initial_states):
        for ep in range(n_episodes_per_initial_state):

            seed = seed + 1
            rng = np.random.default_rng(seed)
            policy = DMatchPolicy(
                rng=rng,
                ou_theta=0.05,
                ou_sigma_A=0.20,
                ou_sigma_B=0.25,
                ou_sigma_C=0.15,
                kick_steps=(10, 20),
                kick_flip_prob=0.25,
                action_smooth=0.95,
            )

            if is_rgb:
                obs_pixel = env.reset(initial_state=initial_state)
                obs = env.get_state_obs()
            else:
                obs = env.reset(initial_state=initial_state)

            policy.reset_episode()

            ep_transitions = []
            for t in range(max_steps):
                act = policy.act(obs)  # behavior action
                act = torch.from_numpy(act)
                if is_rgb:
                    obs_next_pixel, reward, done, info = env.step(act)
                    obs_next = env.get_state_obs()
                else:
                    obs_next_pixel = None
                    obs_next, reward, done, info = env.step(act)

                ep_transitions.append({
                    # "obs": obs,  # s_t
                    "action": act.numpy(),  # a_t
                    "reward": float(reward) if reward is not None else 0.0,
                    "obs_next": obs_next,  # s_{t+1}
                    "obs_next_pixel": obs_next_pixel,  # s_{t+1}
                })

                if done:
                    break

            episodes.append(ep_transitions)

    return episodes


def run_env_autonomous(env, initial_state, n_steps=100, is_rgb=False):

    obs_pixel_list = []
    obs_list = []
    if is_rgb:
        obs_pixel = env.reset(initial_state=initial_state)
        obs = env.get_state_obs()
        obs_pixel_list.append(obs_pixel)
        obs_list.append(obs)
    else:
        obs = env.reset(initial_state=initial_state)
        obs_list.append(obs)

    for _ in range(n_steps - 1):
        next_obs, reward, done, info = env.step(torch.tensor([0.0]))

        if is_rgb:
            obs_pixel_list.append(next_obs)
            obs_list.append(env.get_state_obs())
        else:
            obs_list.append(next_obs)

    if is_rgb:
        return np.array(obs_pixel_list), np.array(obs_list)
    else:
        return np.array(obs_list)
