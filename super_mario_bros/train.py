"""CLI to start the training of a Super Mario Bros agent."""

import json
from argparse import ArgumentParser, Namespace
from typing import Any

import config
from agent import SuperMarioBros


def cli(args: Any = None) -> Namespace:
    """Parsing the arguments of the CLI."""

    # Instantiate the Parser
    parser = ArgumentParser(
        description="CLI to start the training of a Super Mario Bros agent."
    )

    # Add positional & optional arguments
    parser.add_argument(
        "-tl",
        "--training_location",
        help="Training location of the Agent, either 'local' or 'gcp'",
        nargs="?",
        default=config.TRAINING_LOCATION,
        choices=["local", "gcp"],
        type=str,
    )
    parser.add_argument(
        "-af",
        "--artifact_folder",
        help="Subfolder name to distinguish different training trials",
        nargs="?",
        default=config.ARTIFACT_FOLDER,
        type=str,
    )
    parser.add_argument(
        "-ne",
        "--n_envs",
        help="Number of environments to train in parallel",
        nargs="?",
        default=config.N_ENVS,
        type=int,
    )
    parser.add_argument(
        "-ek",
        "--env_kwargs",
        help="Optional keyword arguments to pass to the env constructor",
        nargs="?",
        default=config.ENV_KWARGS,
        type=json.loads,
    )
    parser.add_argument(
        "-p",
        "--policy",
        help="The policy model to use (MlpPolicy, CnnPolicy, ...)",
        nargs="?",
        default=config.POLICY,
        choices=["MlpPolicy", "CnnPolicy"],
        type=str,
    )
    parser.add_argument(
        "-sf",
        "--save_freq",
        help="Save checkpoints every 'save_freq' call of the callback",
        nargs="?",
        default=config.SAVE_FREQ,
        type=int,
    )
    parser.add_argument(
        "-ef",
        "--eval_freq",
        help="Periodically evaluation of current agent performance",
        nargs="?",
        default=config.EVAL_FREQ,
        type=int,
    )
    parser.add_argument(
        "-tt",
        "--total_timesteps",
        help="Total number of samples (env steps) to train on",
        nargs="?",
        default=config.TOTAL_TIMESTEPS,
        type=int,
    )
    parser.add_argument(
        "-ns",
        "--n_steps",
        help="Number of steps to run for each environment per update",
        nargs="?",
        default=config.N_STEPS,
        type=int,
    )
    parser.add_argument(
        "-pk",
        "--policy_kwargs",
        help="Additional arguments to be passed to the policy on creation",
        nargs="?",
        default=config.POLICY_KWARGS,
        type=json.loads,
    )
    parser.add_argument(
        "-lr",
        "--learning_rate",
        help="Learning rate, it can be a function of the current progress remaining (from 1 to 0)",
        nargs="?",
        default=config.LEARNING_RATE,
        type=float,
    )
    parser.add_argument(
        "-g",
        "--gamma",
        help="Discount factor",
        nargs="?",
        default=config.GAMMA,
        type=float,
    )

    return parser.parse_args(args)


def main(args: Namespace) -> bool:
    """Setup and start agent training.

    Args:
        args (object): Parsed CLI arguments.
    """
    # Argument fallbacks
    if isinstance(args.env_kwargs, dict):
        if not args.env_kwargs:
            args.env_kwargs = None

    if isinstance(args.policy_kwargs, dict):
        if not args.policy_kwargs:
            args.policy_kwargs = None

    # Init environment & start training
    env = SuperMarioBros(
        n_envs=args.n_envs,
        env_kwargs=args.env_kwargs,
    )
    env.train(
        training_location=args.training_location,
        artifact_folder=args.artifact_folder,
        policy=args.policy,
        save_freq=args.save_freq,
        eval_freq=args.eval_freq,
        total_timesteps=args.total_timesteps,
        n_steps=args.n_steps,
        policy_kwargs=args.policy_kwargs,
        learning_rate=args.learning_rate,
        gamma=args.gamma,
    )

    return True


if __name__ == "__main__":
    main(cli())
