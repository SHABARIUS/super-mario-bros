"""Unit tests for super_mario_bros/gcp.py."""

import pytest
from gcp import upload_blob
from google.auth.exceptions import DefaultCredentialsError


def test_upload_blob_raise() -> None:
    """Should raise an exception due to missing credentials."""
    with pytest.raises(DefaultCredentialsError):
        upload_blob("bucket_name", "source", "destination_blob_name", "typ")
