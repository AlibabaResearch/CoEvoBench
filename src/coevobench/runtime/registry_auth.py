"""Docker registry authentication: login/logout around mirror pulls.

Credentials come from the typed config (literal or 'env:NAME'
references). Logins are block-scoped: enter via context manager so the
docker config is never left authenticated after the operation.
"""

from __future__ import annotations

import contextlib
import subprocess
from collections.abc import Iterator

from coevobench.config.schema import RegistryAuth


def _login(auth: RegistryAuth) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            "docker", "login", auth.url,
            "--username", auth.username,
            "--password-stdin",
        ],
        input=auth.password,
        capture_output=True,
        text=True,
        timeout=120,
    )


def _logout(url: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["docker", "logout", url], capture_output=True, text=True, timeout=120
    )


@contextlib.contextmanager
def registry_login(auth: RegistryAuth | None) -> Iterator[None]:
    """Authenticate for the duration of the block; always restore.

    A missing or partial auth is a no-op (anonymous registry). A
    configured-but-wrong password raises — a secret typo should
    surface, an unset one should not.
    """
    if auth is None or not auth.url or not auth.username:
        yield
        return
    resolved = auth.resolved()
    if not resolved.password:
        raise RuntimeError(f"registry {resolved.url}: username set but password empty")
    result = _login(resolved)
    if result.returncode != 0:
        raise RuntimeError(
            f"docker login failed for {resolved.url}: {result.stderr.strip()[:200]}"
        )
    try:
        yield
    finally:
        _logout(resolved.url)


def mirror_for(image: str, mirrors: list) -> object | None:
    """The longest-prefix matching mirror for an image reference."""
    best = None
    best_len = -1
    for mirror in mirrors:
        if image.startswith(mirror.prefix) and len(mirror.prefix) > best_len:
            best = mirror
            best_len = len(mirror.prefix)
    return best


def tag_source_mirror(mirrors: list, source: str) -> object | None:
    """The first mirror whose tag_source is `source`, if any.

    Registry-keyed mirrors (tag_source="instance_id" for SWE,
    "task_id" for TB) resolve references as <rewrite><key> with no
    probing and no prefix match — the fast path for private
    registries that key evaluation images by task identifier.
    """
    for mirror in mirrors:
        if mirror.tag_source == source and mirror.rewrite:
            return mirror
    return None


@contextlib.contextmanager
def mirror_login(image: str, mirrors: list) -> Iterator[str]:
    """Rewrite the image through its mirror; authenticate while pulling.

    The tag_map (if any) applies before the prefix rewrite, so a
    mirror can bridge tag schemes (e.g. collapsing an env-number tag
    back to the instance-id form). Yields the (possibly rewritten)
    reference and restores anonymous state on exit.
    """
    mirror = mirror_for(image, mirrors)
    if mirror is None:
        yield image
        return
    mapped = mirror.apply_tag_map(image)
    rewritten = mirror.rewrite_ref(mapped)
    if mirror.auth is None:
        yield rewritten
        return
    with registry_login(mirror.auth):
        yield rewritten


__all__ = ["mirror_for", "tag_source_mirror", "mirror_login", "registry_login"]
