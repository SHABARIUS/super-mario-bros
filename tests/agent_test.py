"""Unit tests for super_mario_bros/agent.py."""

import json
import os

import pytest
from agent import write_json
from gym.spaces import Discrete
from numpy import float32, uint8


def test_write_json_use_case(tmp_path) -> None:
    """A temporary directory with a json file should be created."""
    path = f"{tmp_path}/data/"
    file = "summary.json"
    data = {"test": 123}
    write_json(path, file, data)
    assert len(list(tmp_path.iterdir())) == 1
    assert json.load(open(f"{path}/{file}")) == data


def test_supermariobros_init(supermariobros) -> None:
    """All variables should be initialized correctly."""
    assert supermariobros.n_envs == 1
    assert isinstance(supermariobros.env_kwargs, (dict, type(None))) is True
    assert supermariobros.num_stack == 4
    assert supermariobros.image_resize == 64
    assert supermariobros.observation_space.shape == (4, 64, 64)
    assert supermariobros.action_space == Discrete(2)


def test_supermariobros_step(supermariobros) -> None:
    """A .step() in the environment should return the expected shapes and dtypes."""
    supermariobros.reset()
    observation, reward, done, info = supermariobros.step(
        [supermariobros.action_space.sample()]
    )
    observation.shape == (1, 4, 64, 64)
    observation.dtype == uint8
    reward.shape == (1,)
    reward.dtype == float32
    done.shape == (1,)
    done.dtype == bool
    len(info[0]) == 10


def test_supermariobros_reset(supermariobros) -> None:
    """Should return an initial observation."""
    observation = supermariobros.reset()
    assert observation.shape == (1, 4, 64, 64)
    assert observation.dtype == uint8


@pytest.mark.skip(reason="Calls only the internal function of stable_baseline3.")
def test_supermariobros_render() -> None:
    pass


@pytest.mark.skip(reason="Calls only the internal function of stable_baseline3.")
def test_supermariobros_close() -> None:
    pass


def test_supermariobros_train(supermariobros, tmp_path) -> None:
    """Starts a training attempt and creates artifacts correctly."""
    supermariobros.train(
        training_location="local",
        artifact_folder=f"{tmp_path}/data/",
        save_path="train",
        log_path="logs",
        training_summary=True,
        name_prefix="smb",
        policy="CnnPolicy",
        save_freq=512,
        eval_freq=1024,
        eval_lvl="SuperMarioBros-2-2-v0",
        total_timesteps=1024,
        n_steps=256,
        policy_kwargs=None,
        learning_rate=0.0005,
        gamma=0.98,
        verbose=0,
    )
    summary = {
        "n_envs": 1,
        "env_kwargs": None,
        "joypad_space": [["right"], ["right", "A"]],
        "policy": "CnnPolicy",
        "save_freq": 512,
        "eval_freq": 1024,
        "eval_lvl": "SuperMarioBros-2-2-v0",
        "total_timesteps": 1024,
        "n_steps": 256,
        "policy_kwargs": None,
        "learning_rate": 0.0005,
        "gamma": 0.98,
    }
    assert "data" in os.listdir(tmp_path)
    assert "train" in os.listdir(f"{tmp_path}/data")
    assert "logs" in os.listdir(f"{tmp_path}/data")
    assert "summary.json" in os.listdir(f"{tmp_path}/data")
    assert "smb_512_steps.zip" in os.listdir(f"{tmp_path}/data/train")
    assert "smb_1024_steps.zip" in os.listdir(f"{tmp_path}/data/train")
    assert "evaluations.npz" in os.listdir(f"{tmp_path}/data/logs")
    assert "PPO_1" in os.listdir(f"{tmp_path}/data/logs")
    assert json.load(open(f"{tmp_path}/data/summary.json")) == summary


@pytest.mark.skip(reason="Needs a trained Agent.")
def test_supermariobros_get_prob_dist() -> None:
    pass


@pytest.mark.skip(reason="Would try to render the game.")
def test_supermariobros_play() -> None:
    pass
