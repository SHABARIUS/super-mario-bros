"""Unit tests for super_mario_bros/callbacks.py."""

from callbacks import ArtifactsUploadToGCS


def test_artifactsuploadtogcs_init() -> None:
    """All variables should be initialized correctly."""
    callback = ArtifactsUploadToGCS(
        bucket_name="bucket_name",
        artifact_folder="artifact_folder",
        save_path="save_path",
        log_path=None,
        training_summary_dict=None,
        name_prefix="name_prefix",
        n_envs=3,
        save_freq=100,
    )
    assert callback.bucket_name == "bucket_name"
    assert callback.artifact_folder == "artifact_folder"
    assert callback.save_path == "save_path"
    assert isinstance(callback.log_path, (str, type(None))) is True
    assert isinstance(callback.training_summary_dict, (dict, type(None))) is True
    assert callback.name_prefix == "name_prefix"
    assert callback.n_envs == 3
    assert callback.save_freq == 100
