"""Configuration file for pytest fixtures."""

import pytest
from agent import SuperMarioBros


@pytest.fixture
def supermariobros():
    env = SuperMarioBros(
        n_envs=1,
        env_kwargs=None,
        joypad_space=[["right"], ["right", "A"]],
        num_stack=4,
        image_resize=64,
    )
    return env
