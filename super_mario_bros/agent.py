"""Reinforcement Learning Agent for Super Mario Bros.

The main imported Python packages:
    gym - API for communication between learning algorithms and environments.
    gym_super_mario_bros - OpenAI Gym environment for Super Mario Bros.
    stable_baselines3 - Model free reinforcement learning library based on PyTorch.
"""

import json
import os
import time
import warnings
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, TypeVar, Union

import config
import gym_super_mario_bros
from callbacks import ArtifactsUploadToGCS
from gym import Env
from gym.spaces import Box, Discrete
from gym.wrappers.frame_stack import LazyFrames
from nes_py.wrappers import JoypadSpace
from numpy import uint8
from wrapper import action_wrapper, observation_wrapper, reward_wrapper

# Import into warnings context manager due to TensorBoard DeprecationWarning
with warnings.catch_warnings():

    def warn(*args, **kwargs):
        pass

    warnings.warn = warn
    from stable_baselines3.common.callbacks import (
        CallbackList,
        CheckpointCallback,
        EvalCallback,
    )
    from stable_baselines3.common.env_util import make_vec_env
    from stable_baselines3.common.policies import ActorCriticPolicy
    from stable_baselines3.ppo.ppo import PPO

ObsType = TypeVar("ObsType")
ActType = TypeVar("ActType")


def write_json(path: str, file: str, data: Optional[dict]) -> None:
    """Write data to a json file and create directories as needed.

    Args:
        path (str): Create one or more directories.
        file (str): Name of the json file.
        data (optional dict): Data to dump as a json.
    """
    if data is not None:
        if not os.path.exists(path):
            os.makedirs(path)

        with open(os.path.join(path, file), "w") as f:
            json.dump(data, f)


class SuperMarioBros(Env):
    """Reinforcement Learning Agent for Super Mario Bros."""

    def __init__(
        self,
        n_envs: int = config.N_ENVS,
        env_kwargs: Union[Dict[str, Any], None] = config.ENV_KWARGS,
        joypad_space: List[List[str]] = config.JOYPAD_SPACE,
        num_stack: int = config.NUM_STACK,
        image_resize: int = config.IMAGE_RESIZE,
    ) -> None:
        """Reinforcement Learning Agent for Super Mario Bros.

        Args:
            n_envs (int): Number of environments to train in parallel.
            env_kwargs (optional dict): Optional keyword arguments to pass to the env constructor.
                i.e. With {"stages": ['1-2', '1-2', '1-3', '1-4']} the interactions take place only at certain levels.
            joypad_space (list): Allowed action space.
            num_stack (int): Number of stacked observations.
            image_resize (int): The resized observation, for width and height.
        """
        super().__init__()
        self.n_envs = n_envs
        self.env_kwargs = env_kwargs
        self.joypad_space = joypad_space
        self.num_stack = num_stack
        self.image_resize = image_resize

        # Specify the observation and action space
        self.observation_space = Box(
            low=0,
            high=255,
            shape=(self.num_stack, self.image_resize, self.image_resize),
            dtype=uint8,
        )
        self.action_space = Discrete(2)

        # Initialize the parallel environments
        self.env = make_vec_env(
            "SuperMarioBrosRandomStages-v0",
            n_envs=self.n_envs,
            wrapper_class=self._environment_wrapper,
            env_kwargs=self.env_kwargs,
        )

    def _environment_wrapper(self, env: Env) -> Env:
        """Collection of environment wrapper for agent training.

        Args:
            env (Env): OpenAI Gym environment.
        """
        # Limit the action-space, alternatively use `SIMPLE_MOVEMENT`
        env = JoypadSpace(env, self.joypad_space)

        # Add custom action, observation and reward wrappers
        env = observation_wrapper(env)
        env = reward_wrapper(env)
        # env = action_wrapper(env)
        return env

    def step(
        self, action: ActType
    ) -> Tuple[ObsType, List[float], List[bool], List[Dict]]:
        """Run one timestep of the environment's dynamics.

        This Agent uses an old version of gym (0.21), in a newer version `step`
        will return a tuple `(observation, reward, terminated, truncated, info)`.

        Args:
            action (ActType): An action provided by the agent.

        Returns:
            observation (object): This will be an element of the environment's :attr:`observation_space`.
            reward (float): The amount of reward returned as a result of taking the action.
            done (bool): A boolean value for if the episode has ended.
            info (dictionary): Contains auxiliary diagnostic information (helpful for debugging, learning, and logging).
        """
        observation, reward, done, info = self.env.step(action)
        return observation, reward, done, info

    def reset(self) -> Union[LazyFrames, Any]:
        """Reset environment state and returns the initial observation."""
        return self.env.reset()

    def render(self, *args, **kwargs) -> None:
        """Render the game."""
        self.env.render()

    def close(self) -> None:
        """Close the game environment."""
        self.env.close()

    def train(
        self,
        training_location: str = config.TRAINING_LOCATION,
        bucket_name: str = config.BUCKET_NAME,
        artifact_folder: Optional[str] = config.ARTIFACT_FOLDER,
        save_path: str = config.SAVE_PATH,
        log_path: Optional[str] = config.LOG_PATH,
        training_summary: bool = config.TRAINING_SUMMARY,
        name_prefix: str = config.NAME_PREFIX,
        policy: Union[str, Type[ActorCriticPolicy]] = config.POLICY,
        save_freq: int = config.SAVE_FREQ,
        eval_freq: int = config.EVAL_FREQ,
        eval_lvl: str = config.EVAL_LVL,
        total_timesteps: int = config.TOTAL_TIMESTEPS,
        n_steps: int = config.N_STEPS,
        policy_kwargs: Optional[Dict[str, Any]] = config.POLICY_KWARGS,
        learning_rate: Union[float, Callable[[float], float]] = config.LEARNING_RATE,
        gamma: float = config.GAMMA,
        verbose: int = config.VERBOSE,
    ) -> None:
        """Starts the training of an Agent.

        Args:
            training_location (str): Training location of the Agent, either `local` or `gcp`.
            bucket_name (str): ID of the GCS bucket.
            artifact_folder (optional str): Subfolder name to distinguish different training trials.
            save_path (str): Path to the folder where the model will be saved.
            log_path (optional str): Log location for tensorboard and evaluations (if None, no logging).
            training_summary (bool): True, creates a dictionary with the most important training parameters.
            name_prefix (str): Name prefix for the saved model.
            policy (object): The policy model to use (MlpPolicy, CnnPolicy, ...).
            save_freq (int): Save checkpoints every `save_freq` call of the callback.
            eval_freq (int): Periodically evaluation of current agent performance.
            eval_lvl (str): Level to run the evaluation on.
            total_timesteps (int): Total number of samples (env steps) to train on.
            n_steps (int): Number of steps to run for each environment per update.
            policy_kwargs (optional object): Additional arguments to be passed to the policy on creation.
                Define a custom actor (pi) and value function (vf) for the network architecture.
                i.e. policy_kwargs={'net_arch': [dict(pi=[128, 256, 512], vf=[256, 512, 1024])]}
            learning_rate (float): It can also be a function of the current progress remaining (from 1 to 0).
            gamma (float): Discount factor.
            verbose (int): 0 for no output, 1 for info messages, 2 for debug messages.
        """
        # Creates a json file for later analysis of the training parameters
        training_summary_dict: Optional[dict] = None
        if training_summary:
            training_summary_dict = {
                "n_envs": self.n_envs,
                "env_kwargs": self.env_kwargs,
                "joypad_space": self.joypad_space,
                "policy": policy,
                "save_freq": save_freq,
                "eval_freq": eval_freq,
                "eval_lvl": eval_lvl,
                "total_timesteps": total_timesteps,
                "n_steps": n_steps,
                "policy_kwargs": policy_kwargs,
                "learning_rate": learning_rate,
                "gamma": gamma,
            }

        # Creates a timestamp based artifacts folder name
        if artifact_folder is None:
            artifact_folder = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Upload model artifacts to Cloud Storage or a local folder
        if training_location == "gcp":
            checkpoint_callback = ArtifactsUploadToGCS(
                bucket_name=bucket_name,
                artifact_folder=artifact_folder,
                save_path=save_path,
                log_path=log_path,
                training_summary_dict=training_summary_dict,
                n_envs=self.n_envs,
                save_freq=save_freq // self.n_envs,
            )

        elif training_location == "local":
            # Create training summary & adjust paths for artifact folders
            write_json(artifact_folder, "summary.json", training_summary_dict)
            save_path = f"{artifact_folder}/{save_path}"
            log_path = f"{artifact_folder}/{log_path}"

            checkpoint_callback = CheckpointCallback(
                save_freq=save_freq // self.n_envs,
                save_path=save_path,
                name_prefix=name_prefix,
                save_replay_buffer=True,
                save_vecnormalize=True,
            )

        else:
            raise Exception("Wrong training_location, use either `local` or `gcp`!")

        # Periodically evaluation of current agent performance in a separate environment
        eval_env = gym_super_mario_bros.make(eval_lvl)
        eval_env = self._environment_wrapper(eval_env)
        eval_callback = EvalCallback(
            eval_env,
            eval_freq=eval_freq // self.n_envs,
            log_path=log_path,
            deterministic=False,
            render=False,
            verbose=1,
        )

        # Wrap multiple callbacks
        callback = CallbackList([checkpoint_callback, eval_callback])

        # Instantiate a PPO model
        model = PPO(
            policy,
            self.env,
            tensorboard_log=log_path,
            n_steps=n_steps // self.n_envs,
            policy_kwargs=policy_kwargs,
            learning_rate=learning_rate,
            gamma=gamma,
            verbose=verbose,
        )

        # Start the agent training
        model.learn(total_timesteps=total_timesteps, callback=callback)

    def _get_prob_dist(
        self, model: PPO, obs: Union[LazyFrames, Any]
    ) -> List[List[int]]:
        """Get the action probability distribution from a given observation.

        Args:
            model (object): A trained stable_baselines3 model.
            obs (object): Gym environment observation.
        """
        # obs = obs_as_tensor(state, model.policy.device)
        obs = model.policy.obs_to_tensor(obs.__array__())[0]
        dist = model.policy.get_distribution(obs)
        probs = dist.distribution.probs
        return probs.detach().numpy()

    def play(
        self,
        path: Optional[str] = None,
        deterministic: bool = False,
        print_prob_dist: bool = False,
    ) -> None:
        """Agent playing the game.

        Args:
            path (optional str): If `path` is None an agent will interact randomly with the environment.
            deterministic (bool): False, the model will use the probabilities to return a prediction.
                True, the model is going to return always the best action.
            print_prob_dist (bool): True, will print the action probability distribution from a given observation.
        """
        # Load a trained model
        model: Optional[PPO] = None
        if isinstance(path, str):
            model = PPO.load(path)

        observation = self.reset()
        while True:
            time.sleep(0.01)

            # Agent use random actions for each environment
            if model is None:
                actions = []
                for _ in range(self.n_envs):
                    actions.append(self.env.action_space.sample())
                observation, reward, done, info = self.step(actions)

            # Agent performs actions for each environment according to its trained policy
            elif isinstance(model, PPO):
                action, _ = model.predict(
                    observation.__array__(), deterministic=deterministic
                )
                observation, reward, done, info = self.step(action)

                # if print_prob_dist:
                #     for _ in range(observation.shape[0]):
                #         print(self._get_prob_dist(model, observation))

            self.render()
            if all(done):
                print(f"Info: {info}")
                break

        self.close()
