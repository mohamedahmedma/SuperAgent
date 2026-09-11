"""The ceiling a knowledge-base upload actually meets.

Uploading a 2 MB PDF answered 413 before any application code ran (2026-09-12). Nothing
in `backend/` imposes a limit — `save_upload_file` streams the body a megabyte at a time
and never sizes it — and the public API vhost allowed 100m. The refusal came from
`frontend/nginx.conf`, which said nothing about body size at all, so nginx applied its
built-in default of 1m.

That is the shape of failure worth a permanent test: not a wrong value somebody can read,
but an ABSENT directive whose default is invisible. The config looked complete, the limit
was never written down anywhere, and the 413 carried no clue about which of the three
hops in front of the backend had produced it.

It is easy to reintroduce, because which hop decides depends on how the frontend was
built. `frontend/Dockerfile` takes only `VITE_IDENTITY_BASE_URL` as a build arg, so
`VITE_API_BASE_URL` is empty in the image and `utils/api.ts` falls back to the page's own
origin: uploads go through the frontend container. A build that did bake the API base URL
in would send them to api.aurexis.cc instead. Both paths have to allow the same size, and
whichever is lower is the one an admin meets — so this asserts the floor on every vhost
that fronts the backend, found by reading the configs rather than by listing them.
"""
import io
import re
import unittest
from pathlib import Path

from starlette.applications import Starlette
from starlette.responses import JSONResponse
from starlette.routing import Route
from starlette.testclient import TestClient

REPO_ROOT = Path(__file__).resolve().parents[2]

#: What a deployment must accept. The knowledge base ingests scanned manuals, which run to
#: hundreds of megabytes; the number is the product decision, this file only enforces it.
MINIMUM_UPLOAD_BYTES = 500 * 1024 * 1024

#: The backend's port, in both forms the configs use to name it: a container address and a
#: loopback address. A vhost that proxies here can carry `/documents`.
BACKEND_UPSTREAMS = ("backend:8000", "127.0.0.1:8000")

_SIZE = re.compile(r"client_max_body_size\s+(\d+)\s*([kmg]?)\s*;", re.IGNORECASE)
_SUFFIX = {"": 1, "k": 1024, "m": 1024 * 1024, "g": 1024 * 1024 * 1024}


def _nginx_configs() -> list[Path]:
    """Every nginx config this repository ships.

    Discovered, not listed, so a vhost added later is covered the day it lands rather than
    the day somebody remembers this file. `volumes/` is excluded: it holds postgres's
    runtime `.conf` files, which are not nginx's and share only the extension.
    """
    found = [REPO_ROOT / "frontend" / "nginx.conf"]
    found += sorted((REPO_ROOT / "deploy" / "nginx").glob("*.conf"))
    return [path for path in found if path.is_file()]


def _strip_comments(text: str) -> str:
    """Drop `#` comments so prose about sizes is never read as configuration.

    The comments in these files discuss `1m` defaults and 512 MB ceilings by name, and a
    regex over the raw text would happily parse one of those sentences as the directive.
    """
    return "\n".join(line.split("#", 1)[0] for line in text.splitlines())


def _declared_limit(path: Path) -> int | None:
    """The vhost's `client_max_body_size` in bytes, or None when it declares none.

    None is the interesting answer: it is what nginx reads as 1m, and what this whole file
    exists to catch.
    """
    match = _SIZE.search(_strip_comments(io.open(path, encoding="utf-8").read()))
    if match is None:
        return None
    return int(match.group(1)) * _SUFFIX[match.group(2).lower()]


class UploadCeilingTests(unittest.TestCase):
    def test_every_vhost_fronting_the_backend_allows_a_large_upload(self):
        checked = []
        for path in _nginx_configs():
            body = _strip_comments(io.open(path, encoding="utf-8").read())
            if not any(upstream in body for upstream in BACKEND_UPSTREAMS):
                continue  # identity only; sign-in payloads are kilobytes.
            checked.append(path)
            limit = _declared_limit(path)
            relative = path.relative_to(REPO_ROOT).as_posix()

            self.assertIsNotNone(
                limit,
                f"{relative} proxies to the backend but declares no client_max_body_size, "
                "so nginx applies its 1m default and a knowledge-base upload is refused "
                "with a 413 no application log can explain.",
            )
            self.assertGreaterEqual(
                limit,
                MINIMUM_UPLOAD_BYTES,
                f"{relative} caps uploads at {limit} bytes, below the "
                f"{MINIMUM_UPLOAD_BYTES} the knowledge base must accept.",
            )

        # Guards the discovery itself: a rename that matched nothing would otherwise let
        # this test pass by checking no files at all.
        self.assertEqual(
            len(checked),
            2,
            "expected the frontend proxy and the public API vhost to front the backend, "
            f"found {[p.relative_to(REPO_ROOT).as_posix() for p in checked]}",
        )

    def test_the_two_hops_agree(self):
        """An admin traverses one hop or the other, never both, and cannot tell which.

        Letting them drift means an upload that succeeds from one build and 413s from
        another, with nothing in the UI to distinguish the two.
        """
        limits = {
            path.relative_to(REPO_ROOT).as_posix(): _declared_limit(path)
            for path in _nginx_configs()
            if any(u in _strip_comments(io.open(path, encoding="utf-8").read()) for u in BACKEND_UPSTREAMS)
        }
        self.assertEqual(
            len(set(limits.values())),
            1,
            f"the hops in front of the backend disagree on the upload ceiling: {limits}",
        )


class StarletteFilePartTests(unittest.TestCase):
    """The other half of the claim, which is a dependency's behaviour rather than ours.

    Raising the nginx ceiling only helps if nothing behind it caps the body too, and
    Starlette does carry a 1 MB `max_part_size`. It applies to FIELDS: `on_part_data`
    checks it only under `if self._current_part.file is None`, while a file part is
    streamed to a `SpooledTemporaryFile` whose own 1 MB `spool_max_size` merely decides
    when it stops living in memory. So a large FILE is fine and a large TEXT field is not.

    That is a fine distinction resting on an implementation detail, and if a future
    Starlette applied the limit uniformly, uploads would break at 1 MB again — with the
    nginx configs above looking entirely correct. Hence a test that exercises the real
    parser rather than a comment asserting what it does.
    """

    @staticmethod
    def _client() -> TestClient:
        async def endpoint(request):
            form = await request.form()
            upload = form["file"]
            return JSONResponse({"size": len(await upload.read())})

        return TestClient(Starlette(routes=[Route("/u", endpoint, methods=["POST"])]))

    def test_a_file_part_far_above_one_megabyte_is_parsed(self):
        payload = b"x" * (5 * 1024 * 1024)
        response = self._client().post(
            "/u", files={"file": ("big.pdf", payload, "application/pdf")}
        )
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json()["size"], len(payload))


if __name__ == "__main__":
    unittest.main()
