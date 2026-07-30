"""Content-addressed blob storage for raw asset bytes (ports and adapters).

Storage is keyed by sha256, never by filename. Three properties fall out of that and
all three matter in production:

* **Idempotent writes** — re-uploading a document rewrites nothing.
* **Automatic deduplication** — one copy of a logo, no matter how many documents
  embed it.
* **Safe paths** — a sha256 is a fixed-length hex string, so no filename from a user
  document ever reaches the filesystem or a bucket key.

`LocalBlobStore` is the default and needs no infrastructure. `S3BlobStore` targets the
MinIO already running for Milvus (its own bucket, never Milvus's) and imports boto3
lazily, so the dependency is only required by deployments that select it.
"""
from __future__ import annotations

import logging
import os
import shutil
import tempfile
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# content-type -> extension. Extensions are cosmetic (they make a blob directory
# browsable); the sha256 is the identity.
_EXTENSIONS = {
    "image/png": ".png",
    "image/jpeg": ".jpg",
    "image/jpg": ".jpg",
    "image/webp": ".webp",
    "image/gif": ".gif",
    "image/tiff": ".tiff",
    "image/bmp": ".bmp",
    "application/pdf": ".pdf",
}
_DEFAULT_EXTENSION = ".bin"


def extension_for(content_type: str) -> str:
    return _EXTENSIONS.get((content_type or "").strip().lower(), _DEFAULT_EXTENSION)


def _validate_digest(sha256: str) -> str:
    """Reject anything that is not a hex digest before it becomes a path or key."""
    digest = (sha256 or "").strip().lower()
    if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
        raise ValueError(f"Invalid sha256 digest: {sha256!r}")
    return digest


def _shard(digest: str, content_type: str) -> str:
    """Two levels of 256-way fan-out. A flat directory of a million blobs is slow to
    list and hostile to most filesystems; this keeps directories small."""
    return f"{digest[:2]}/{digest[2:4]}/{digest}{extension_for(content_type)}"


class BlobStore(ABC):
    """Port. Implementations must be safe to call concurrently and idempotent on put."""

    @abstractmethod
    def put(self, sha256: str, data: bytes, content_type: str = "") -> str:
        """Store bytes, returning the URI. A no-op when the digest already exists."""

    @abstractmethod
    def get(self, uri: str) -> bytes:
        ...

    @abstractmethod
    def exists(self, sha256: str, content_type: str = "") -> bool:
        ...

    @abstractmethod
    def delete(self, uri: str) -> bool:
        """Returns True when something was actually removed."""


class LocalBlobStore(BlobStore):
    """Filesystem adapter. Default for single-node deployments."""

    SCHEME = "file://"

    def __init__(self, root: Path | str):
        self.root = Path(root).resolve()

    def _path_for(self, sha256: str, content_type: str) -> Path:
        return self.root / _shard(_validate_digest(sha256), content_type)

    def _resolve(self, uri: str) -> Path:
        raw = uri[len(self.SCHEME):] if uri.startswith(self.SCHEME) else uri
        path = (self.root / raw).resolve() if not Path(raw).is_absolute() else Path(raw).resolve()
        # Defence in depth: even though keys are digests, never follow a URI out of root.
        if not str(path).startswith(str(self.root)):
            raise ValueError(f"Blob URI escapes the store root: {uri!r}")
        return path

    def put(self, sha256: str, data: bytes, content_type: str = "") -> str:
        path = self._path_for(sha256, content_type)
        uri = f"{self.SCHEME}{path.relative_to(self.root).as_posix()}"
        if path.exists():
            return uri

        path.parent.mkdir(parents=True, exist_ok=True)
        # Write to a temp file in the same directory, then rename: a crash mid-write
        # must never leave a truncated blob at a digest that claims to be complete.
        fd, tmp_name = tempfile.mkstemp(dir=str(path.parent), suffix=".part")
        try:
            with os.fdopen(fd, "wb") as handle:
                handle.write(data)
            os.replace(tmp_name, path)
        except BaseException:
            Path(tmp_name).unlink(missing_ok=True)
            raise
        return uri

    def get(self, uri: str) -> bytes:
        return self._resolve(uri).read_bytes()

    def exists(self, sha256: str, content_type: str = "") -> bool:
        return self._path_for(sha256, content_type).exists()

    def delete(self, uri: str) -> bool:
        path = self._resolve(uri)
        if not path.exists():
            return False
        path.unlink()
        # Prune the now-empty shard directories so the tree does not accumulate
        # hundreds of thousands of empty folders over a corpus's lifetime.
        for parent in (path.parent, path.parent.parent):
            try:
                if parent != self.root and not any(parent.iterdir()):
                    parent.rmdir()
            except OSError:
                break
        return True


class S3BlobStore(BlobStore):
    """S3/MinIO adapter. boto3 is imported lazily so it stays an optional dependency."""

    SCHEME = "s3://"

    def __init__(
        self,
        bucket: str,
        endpoint_url: Optional[str] = None,
        access_key: Optional[str] = None,
        secret_key: Optional[str] = None,
        region: Optional[str] = None,
    ):
        self.bucket = bucket
        self._endpoint_url = endpoint_url
        self._access_key = access_key
        self._secret_key = secret_key
        self._region = region
        self._client = None

    def _get_client(self):
        if self._client is None:
            try:
                import boto3  # noqa: PLC0415 — optional dependency, imported on use
            except ImportError as exc:
                raise RuntimeError(
                    "S3 blob storage requires boto3. Install it, or set the profile's "
                    "assets.blob_backend to 'local'."
                ) from exc
            self._client = boto3.client(
                "s3",
                endpoint_url=self._endpoint_url,
                aws_access_key_id=self._access_key,
                aws_secret_access_key=self._secret_key,
                region_name=self._region,
            )
        return self._client

    def _key_for(self, sha256: str, content_type: str) -> str:
        return _shard(_validate_digest(sha256), content_type)

    def _key_from_uri(self, uri: str) -> str:
        raw = uri[len(self.SCHEME):] if uri.startswith(self.SCHEME) else uri
        prefix = f"{self.bucket}/"
        return raw[len(prefix):] if raw.startswith(prefix) else raw

    def put(self, sha256: str, data: bytes, content_type: str = "") -> str:
        key = self._key_for(sha256, content_type)
        uri = f"{self.SCHEME}{self.bucket}/{key}"
        if self.exists(sha256, content_type):
            return uri
        extra = {"ContentType": content_type} if content_type else {}
        self._get_client().put_object(Bucket=self.bucket, Key=key, Body=data, **extra)
        return uri

    def get(self, uri: str) -> bytes:
        response = self._get_client().get_object(Bucket=self.bucket, Key=self._key_from_uri(uri))
        return response["Body"].read()

    def exists(self, sha256: str, content_type: str = "") -> bool:
        try:
            self._get_client().head_object(Bucket=self.bucket, Key=self._key_for(sha256, content_type))
            return True
        except Exception:
            return False

    def delete(self, uri: str) -> bool:
        key = self._key_from_uri(uri)
        try:
            self._get_client().delete_object(Bucket=self.bucket, Key=key)
            return True
        except Exception:
            logger.exception("Failed to delete blob %s", uri)
            return False


_store: Optional[BlobStore] = None


def build_blob_store(assets_config, project_root: Optional[Path] = None) -> BlobStore:
    """Construct the adapter the profile selects. Credentials come from the
    environment, never from the profile — profiles are committed to the repository."""
    backend = (assets_config.blob_backend or "local").strip().lower()

    if backend == "local":
        root = Path(assets_config.blob_root)
        if not root.is_absolute():
            base = project_root or Path(__file__).resolve().parents[2]
            root = base / root
        return LocalBlobStore(root)

    if backend == "s3":
        return S3BlobStore(
            bucket=assets_config.blob_bucket,
            endpoint_url=os.getenv("ASSET_S3_ENDPOINT") or os.getenv("MINIO_ENDPOINT"),
            access_key=os.getenv("ASSET_S3_ACCESS_KEY") or os.getenv("MINIO_ACCESS_KEY"),
            secret_key=os.getenv("ASSET_S3_SECRET_KEY") or os.getenv("MINIO_SECRET_KEY"),
            region=os.getenv("ASSET_S3_REGION"),
        )

    raise ValueError(f"Unknown assets.blob_backend: {backend!r} (expected 'local' or 's3')")


def get_blob_store() -> BlobStore:
    global _store
    if _store is None:
        from backend.profiles import get_profile

        _store = build_blob_store(get_profile().assets)
    return _store


def set_blob_store(store: Optional[BlobStore]) -> None:
    """Swap the process-wide store. For tests and for the backfill CLI."""
    global _store
    _store = store


def purge_local_store(root: Path | str) -> None:
    """Remove an entire local blob tree. Destructive; used only by test teardown."""
    shutil.rmtree(Path(root), ignore_errors=True)
