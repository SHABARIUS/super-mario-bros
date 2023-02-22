"""Collection of observation preprocessing wrapper.

There is more to Wrappers than the vanilla Wrapper class.
Gym also provides you with specific wrappers that target specific elements of the environment,
such as observations, rewards, and actions. Their use is demonstrated in the following section.

ObservationWrapper: This helps us make changes to the observation using the observation method of the wrapper class.
RewardWrapper: This helps us make changes to the reward using the reward function of the wrapper class.
ActionWrapper: This helps us make changes to the action using the action function of the wrapper class.
"""

import random
from typing import Any, List, Tuple, TypeVar, Union

import config
from gym.core import ActionWrapper, Env, ObservationWrapper, RewardWrapper, Wrapper
from gym.wrappers.frame_stack import FrameStack
from gym.wrappers.gray_scale_observation import GrayScaleObservation
from gym.wrappers.resize_observation import ResizeObservation
from gym.wrappers.transform_observation import TransformObservation
from numpy import ndarray

ObsType = TypeVar("ObsType", float, ndarray)
ActType = TypeVar("ActType")


# Collection of action wrapper
class RandomActionWrapper(ActionWrapper):
    """Replace the current action with a random one."""

    def __init__(self, env: Env, epsilon: float = 0.1) -> None:
        """Replace the current action with a random one.

        By issuing the random actions, the agent explore the environment
        and deviate from the beaten path of its strategy from time to time.

        Args:
            env (Env): OpenAI Gym environment.
            epsilon (float): Probability value.
        """
        super().__init__(env)
        self.epsilon = epsilon

    def action(self, action: ActType) -> ActType:
        """Modify the action before `env.step()`.

        Args:
            action (ActType): An action provided by the agent.
        """
        if random.random() < self.epsilon:
            return self.env.action_space.sample()
        return action


def action_wrapper(env: Env) -> Env:
    """Apply all action wrapper.

    Args:
        env (Env): OpenAI Gym environment.
    """
    return RandomActionWrapper(env)


# Collection of basic and observation wrapper
class SkipFrame(Wrapper):
    """Return only every `skip`-th frame."""

    def __init__(self, env: Env, skip: int) -> None:
        """Return only every `skip`-th frame.

        Args:
            env (Env): OpenAI Gym environment.
            skip (int): Every `skip`-th frame to be returned.
        """
        super().__init__(env)
        self._skip = skip

    def step(self, action: ActType) -> Tuple[ObsType, float, bool, dict]:
        """Repeat action and sum rewards.

        Args:
            action (ActType): An action provided by the agent.
        """
        total_reward = 0.0
        done = False
        for i in range(self._skip):
            observation, reward, done, info = self.env.step(action)
            total_reward += reward
            if done:
                break
        return observation, total_reward, done, info


class NormalizeObservation(ObservationWrapper):
    """Normalize observation."""

    def __init__(self, env: Env) -> None:
        """Normalize observation.

        Args:
            env (Env): OpenAI Gym environment.
        """
        super().__init__(env)

    def observation(self, obs: ObsType) -> ObsType:
        return obs / 255.0


def observation_wrapper(env: Env) -> FrameStack:
    """Apply a preprocessing wrapper to the observations.

    No need for normalization, because the PPO('CnnPolicy') does that automatically.

    skip - Return only every `skip`-th frame.
    crop - Crop the image from (240, 256, 3) to (185, 256, 3).
    resize - Resize the image from (185, 256, 3) to (64, 64, 3).
    gray - Reduce the color layer from (64, 64, 3) to (64, 64).
    stack - Rolling frame stacking from (64, 64) to (4, 64, 64).

    Args:
        env (Env): OpenAI Gym environment.
    """
    skip = SkipFrame(env, skip=config.SKIP)
    crop = TransformObservation(skip, lambda obs: obs[45:-10, :])
    resize = ResizeObservation(crop, shape=config.IMAGE_RESIZE)
    gray = GrayScaleObservation(resize, keep_dim=False)
    # norm = NormalizeObservation(gray)
    return FrameStack(gray, num_stack=config.NUM_STACK)


# Collection of reward wrapper
class ScaleRewards(RewardWrapper):
    """Scale all rewards by a factor."""

    def __init__(self, env: Env, factor: float = 0.01) -> None:
        """Scale all rewards by a factor.

        Args:
            env (Env): OpenAI Gym environment.
            factor (float): Scaling factor
        """
        super().__init__(env)
        self.factor = factor

    def reward(self, reward: Union[int, float]) -> Union[int, float]:
        """Modify the returning reward from a `env.step()`.

        Args:
            reward (int float): Reward that is returned by the base environment.
        """
        return reward * self.factor


def reward_wrapper(env: Env) -> Env:
    """Apply all reward wrapper.

    Args:
        env (Env): OpenAI Gym environment.
    """
    return ScaleRewards(env)
