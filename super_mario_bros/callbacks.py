"""Collection of training callbacks.

A callback is a set of functions that will be called at given stages of the training procedure.
You can use callbacks to access internal state of the RL model during training.
It allows one to do monitoring, auto saving, model manipulation, progress bars, …

Documentation: https://stable-baselines3.readthedocs.io/en/master/guide/callbacks.html
"""

import glob
import json
import os
import warnings
from typing import Any, Dict, Optional

import config
from gcp import upload_blob

# Import into warnings context manager due to TensorBoard DeprecationWarning
with warnings.catch_warnings():

    def warn(*args, **kwargs):
        pass

    warnings.warn = warn
    from stable_baselines3.common.callbacks import BaseCallback


class ArtifactsUploadToGCS(BaseCallback):
    """Upload the model artifacts to Cloud Storage at each call."""

    def __init__(
        self,
        bucket_name: str = config.BUCKET_NAME,
        artifact_folder: str = "artifacts",
        save_path: str = config.SAVE_PATH,
        log_path: Optional[str] = None,
        training_summary_dict: Optional[Dict[str, Any]] = None,
        name_prefix: str = config.NAME_PREFIX,
        n_envs: int = config.N_ENVS,
        save_freq: int = config.SAVE_FREQ,
        verbose: int = 0,
    ) -> None:
        """Upload the model artifacts to Cloud Storage at each call.

        Artifacts are a trained model, tensorboard logs & training summary.

        Args:
            bucket_name (str): ID of the GCS bucket.
            artifact_folder (str): Subfolder name to distinguish different training trials.
            save_path (str): Local folder path where the artifacts will be saved.
            log_path (optional str): Log location for tensorboard and evaluations (if None, no logging).
            training_summary_dict (optional dict): Uploads a dictionary as summary.json to GCS.
            name_prefix (str): Name prefix for the saved model.
            n_envs (int): Number of environments to train in parallel.
            save_freq (int): Save checkpoints every `save_freq` call of the callback.
            verbose (int): 0 for no output, 1 for info messages, 2 for debug messages.
        """
        super(ArtifactsUploadToGCS, self).__init__(verbose)
        self.bucket_name = bucket_name
        self.artifact_folder = artifact_folder
        self.save_path = save_path
        self.log_path = log_path
        self.training_summary_dict = training_summary_dict
        self.name_prefix = name_prefix
        self.n_envs = n_envs
        self.save_freq = save_freq

    def _init_callback(self) -> None:
        """Creates a local artifacts folder."""
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_training_start(self) -> None:
        """This method is called before the first rollout starts."""

        # Upload the training summary to GCS
        if isinstance(self.training_summary_dict, dict):
            with open("summary.json", "w") as outfile:
                json.dump(self.training_summary_dict, outfile)

            source = "summary.json"
            destination_blob_name = f"{self.artifact_folder}/{source}"
            upload_blob(self.bucket_name, source, destination_blob_name)

    def _on_step(self) -> None:
        """This method will be called by the model after each call to `env.step()`."""
        if self.n_calls % self.save_freq == 0:
            model_path = os.path.join(
                self.save_path, f"{self.name_prefix}_{self.n_calls * self.n_envs}"
            )
            self.model.save(model_path)

            # Upload the trained model, tensorboard logs and evaluations to GCS
            source = f"{model_path}.zip"
            destination_blob_name = f"{self.artifact_folder}/train/{self.name_prefix}_{self.n_calls * self.n_envs}.zip"
            upload_blob(self.bucket_name, source, destination_blob_name)

            if self.log_path is not None:
                source = glob.glob(f"{config.LOG_PATH}/PPO_*/*")[0]
                blob_name = source.split("/")[-1]
                destination_blob_name = f"{self.artifact_folder}/logs/{blob_name}"
                upload_blob(self.bucket_name, source, destination_blob_name)

                source = glob.glob(f"{config.LOG_PATH}/evaluations.npz")[0]
                blob_name = source.split("/")[-1]
                destination_blob_name = f"{self.artifact_folder}/logs/{blob_name}"
                upload_blob(self.bucket_name, source, destination_blob_name)
