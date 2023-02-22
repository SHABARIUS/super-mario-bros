"""Unit tests for super_mario_bros/wrapper.py."""

from agent import SuperMarioBros
from wrapper import NormalizeObservation, RandomActionWrapper, ScaleRewards, SkipFrame


def test_wrapper_init() -> None:
    """All wrappers should be correctly initialized."""
    env = SuperMarioBros(n_envs=1, num_stack=4, image_resize=64)
    env = SkipFrame(env, skip=4)
    env = NormalizeObservation(env)
    env = ScaleRewards(env)
    env = RandomActionWrapper(env)
    assert env.class_name() == "RandomActionWrapper"
    assert env.epsilon == 0.1
