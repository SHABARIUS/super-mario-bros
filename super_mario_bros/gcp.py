"""Functions for interaction with the Google Cloud Platform."""

from google.cloud import storage


def upload_blob(
    bucket_name: str, source: str, destination_blob_name: str, typ: str = "filename"
) -> None:
    """Uploads a file to a Cloud Storage bucket.

    Args:
        bucket_name (str): ID of the GCS bucket.
        source (str): Path to the file to upload (filename)
            or the contents to upload to a file (string).
        destination_blob_name (str): ID of the GCS object.
        typ (str): `filename`, will upload a file.
            `string`, will upload content to a file.
    """
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)

    if typ == "filename":
        blob.upload_from_filename(source)

    elif typ == "string":
        blob.upload_from_string(source)

    else:
        raise Exception("Wrong typ, use either `filename` or `string`!")
