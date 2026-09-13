"""The upload area: where an uploaded file lands, and whether this profile accepts it.

What used to be here — the document loader, the parent-chunk store, the Milvus client and
its writer — was constructed at import, so importing any route pulled a vector client and
a chunk store into every process that touched the API. They are built by
`backend/composition.py` now, on first use. The transactional delete that used them is
`backend/indexing/removal.py`.

What remains needs no collaborator: two paths and two questions about a filename.
"""
import os
from pathlib import Path

from backend.profiles import get_profile

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR.parent / "data"
UPLOAD_DIR = DATA_DIR / "documents"


def is_supported_document(filename: str) -> bool:
    """Accepted upload extensions come from the profile: an e-commerce catalogue and a
    document KB do not necessarily ingest the same file types."""
    file_lower = filename.lower()
    extensions = tuple(get_profile().ingest.supported_extensions)
    return bool(extensions) and file_lower.endswith(extensions)


async def save_upload_file(file, file_path: Path) -> None:
    with open(file_path, "wb") as f:
        while True:
            chunk = await file.read(1024 * 1024)
            if not chunk:
                break
            f.write(chunk)


def ensure_upload_dir() -> None:
    os.makedirs(UPLOAD_DIR, exist_ok=True)
