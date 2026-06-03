from typing import Optional
from minio import Minio

from src.config import settings

_client: Optional[Minio] = None


def get_client() -> Minio:
    global _client
    if _client is None:
        _client = Minio(
            settings.minio_endpoint,
            access_key=settings.minio_root_user,
            secret_key=settings.minio_root_password,
            secure=settings.minio_secure,
        )
    return _client


def ensure_buckets() -> None:
    client = get_client()
    for bucket in (settings.minio_bucket_videos, settings.minio_bucket_clips):
        if not client.bucket_exists(bucket):
            client.make_bucket(bucket)
            print(f"Created bucket: {bucket}")


def upload_file(bucket: str, object_key: str, file_path: str, content_type: str = "application/octet-stream") -> None:
    get_client().fput_object(bucket, object_key, file_path, content_type=content_type)


def get_presigned_url(bucket: str, object_key: str, expires_hours: int = 24) -> str:
    from datetime import timedelta
    return get_client().presigned_get_object(bucket, object_key, expires=timedelta(hours=expires_hours))
