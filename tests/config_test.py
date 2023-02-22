"""Unit tests for super_mario_bros/config.py."""

import warnings
from typing import Type

# Import into warnings context manager due to TensorBoard DeprecationWarning
with warnings.catch_warnings():

    def warn(*args, **kwargs):
        pass

    warnings.warn = warn
    import config
    from stable_baselines3.common.policies import ActorCriticPolicy


def test_config_arguments_length() -> None:
    """All variables should be covered in the following tests."""
    variables = len([item for item in dir(config) if not item.startswith("__")])
    assert variables == 26, "There are more or fewer variables than 26"


def test_config_types() -> None:
    """All arguments should have the correct type."""
    assert type(config.TRAINING_LOCATION) == str
    assert type(config.BUCKET_NAME) == str
    assert isinstance(config.ARTIFACT_FOLDER, (str, type(None))) is True
    assert type(config.SAVE_PATH) == str
    assert isinstance(config.LOG_PATH, (str, type(None))) is True
    assert type(config.TRAINING_SUMMARY) == bool
    assert type(config.NAME_PREFIX) == str
    assert type(config.N_ENVS) == int
    assert isinstance(config.ENV_KWARGS, (dict, type(None))) is True
    assert type(config.NUM_STACK) == int
    assert type(config.IMAGE_RESIZE) == int
    assert type(config.SKIP) == int
    assert type(config.POLICY) == str or type(config.POLICY) == Type[ActorCriticPolicy]
    assert type(config.SAVE_FREQ) == int
    assert type(config.EVAL_FREQ) == int
    assert type(config.EVAL_LVL) == str
    assert type(config.TOTAL_TIMESTEPS) == int
    assert type(config.N_STEPS) == int
    assert type(config.BATCH_SIZE) == int
    assert type(config.N_EPOCHS) == int
    assert isinstance(config.POLICY_KWARGS, (dict, type(None))) is True
    assert type(config.LEARNING_RATE) == float
    assert type(config.GAMMA) == float
    assert type(config.VERBOSE) == int
