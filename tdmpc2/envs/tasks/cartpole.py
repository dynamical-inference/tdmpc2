from dm_control.rl import control
from dm_control.suite import cartpole
from dm_control.utils import rewards
import numpy as np

_DEFAULT_TIME_LIMIT = 20


@cartpole.SUITE.add('custom')
def swingup_custom(time_limit=_DEFAULT_TIME_LIMIT,
                   random=None,
                   environment_kwargs=None):
    """Returns cartpole swingup task with custom reward (no cart position penalty)."""
    # Use the standard cartpole physics model
    physics = cartpole.Physics.from_xml_string(*cartpole.get_model_and_assets())
    task = CustomSwingUp(random=random)
    environment_kwargs = environment_kwargs or {}
    return control.Environment(physics,
                               task,
                               time_limit=time_limit,
                               **environment_kwargs)


class CustomSwingUp(cartpole.Balance):
    """A custom Cartpole SwingUp task with modified reward function."""

    def __init__(self, random=None):
        # Initialize as swingup task (swing_up=True) with smooth reward (sparse=False)
        super().__init__(swing_up=True, sparse=False, random=random)

    def get_reward(self, physics):
        """
        Custom reward function for cartpole swingup.
        
        This is identical to the standard cartpole swingup reward, but with the
        'centered' component removed. The reward now only depends on:
        - Pole being upright
        - Small control (penalty for large actions)
        - Small velocity (penalty for high angular velocities)
        
        The cart position is no longer part of the reward, so the pole can be
        upright at any cart position and still get high reward.
        """
        # This matches the standard swingup reward structure from dm_control,
        # but removes the 'centered' component
        upright = (physics.pole_angle_cosine() + 1) / 2
        small_control = rewards.tolerance(physics.control(),
                                          margin=1,
                                          value_at_margin=0,
                                          sigmoid='quadratic')[0]
        small_control = (4 + small_control) / 5
        small_velocity = rewards.tolerance(physics.angular_vel(),
                                           margin=5).min()
        small_velocity = (1 + small_velocity) / 2

        # Removed: * centered (the cart position component)
        return upright.mean() * small_control * small_velocity
