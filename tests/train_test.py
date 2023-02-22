"""Unit tests for super_mario_bros/train.py."""

from argparse import Namespace

import config
import pytest
from train import cli, main


def test_cli_arguments_length() -> None:
    """All arguments should be covered in the following tests."""
    args = cli([])
    assert len(args.__dict__.keys()) == 12


def test_cli_empty_items() -> None:
    """An empty CLI should use all config parameters."""
    args = cli([])
    assert args.training_location == config.TRAINING_LOCATION
    assert args.artifact_folder == config.ARTIFACT_FOLDER
    assert args.n_envs == config.N_ENVS
    assert args.env_kwargs == config.ENV_KWARGS
    assert args.policy == config.POLICY
    assert args.save_freq == config.SAVE_FREQ
    assert args.eval_freq == config.EVAL_FREQ
    assert args.total_timesteps == config.TOTAL_TIMESTEPS
    assert args.n_steps == config.N_STEPS
    assert args.policy_kwargs == config.POLICY_KWARGS
    assert args.learning_rate == config.LEARNING_RATE
    assert args.gamma == config.GAMMA


def test_cli_single_item() -> None:
    """A single CLI argument should only replace its config parameter."""
    args = cli(["-sf", "10000"])
    assert args.training_location == config.TRAINING_LOCATION
    assert args.artifact_folder == config.ARTIFACT_FOLDER
    assert args.n_envs == config.N_ENVS
    assert args.env_kwargs == config.ENV_KWARGS
    assert args.policy == config.POLICY
    assert args.save_freq == 10000
    assert args.eval_freq == config.EVAL_FREQ
    assert args.total_timesteps == config.TOTAL_TIMESTEPS
    assert args.n_steps == config.N_STEPS
    assert args.policy_kwargs == config.POLICY_KWARGS
    assert args.learning_rate == config.LEARNING_RATE
    assert args.gamma == config.GAMMA


def test_cli_all_items() -> None:
    """All CLI arguments should be replaced correctly."""
    args = cli(
        [
            "-tl",
            "local",
            "-af",
            "data",
            "-ne",
            "8",
            "-ek",
            '{"stages": ["1-1", "2-1", "3-1", "4-1"]}',
            "-p",
            "MlpPolicy",
            "-sf",
            "10000",
            "-ef",
            "5000",
            "-tt",
            "5000000",
            "-ns",
            "1024",
            "-pk",
            '{"net_arch":[{"pi":[128,256],"vf":[512,1024]}]}',
            "-lr",
            "0.0005",
            "-g",
            "0.95",
        ]
    )
    assert args.training_location == "local"
    assert args.artifact_folder == "data"
    assert args.n_envs == 8
    assert args.env_kwargs == {"stages": ["1-1", "2-1", "3-1", "4-1"]}
    assert args.policy == "MlpPolicy"
    assert args.save_freq == 10000
    assert args.eval_freq == 5000
    assert args.total_timesteps == 5000000
    assert args.n_steps == 1024
    assert args.policy_kwargs == {"net_arch": [{"pi": [128, 256], "vf": [512, 1024]}]}
    assert args.learning_rate == 0.0005
    assert args.gamma == 0.95


def test_main_raise() -> None:
    """Call main() and start agent training, but raise an exception because of an incorrect training_location."""
    args = Namespace()
    args.training_location = "raises_an_exception"
    args.artifact_folder = None
    args.n_envs = 1
    args.env_kwargs = {}
    args.policy = "CnnPolicy"
    args.save_freq = 512
    args.eval_freq = 1024
    args.total_timesteps = 1024
    args.n_steps = 256
    args.policy_kwargs = {}
    args.learning_rate = 0.0005
    args.gamma = 0.95

    with pytest.raises(Exception):
        main(args=args)
