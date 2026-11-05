"""Operator-only, model-pinned teacher forwarding. Cost accounting is intentionally absent."""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

from coevobench.runtime.run_resources import remaining_seconds

MAX_BODY = 900_000  # Leave room for the enclosing 1 MiB broker JSON message.
MAX_REPLY = 900_000


def chat_completions_url(base_url: str) -> str:
    base = base_url.rstrip("/")
    parsed = urllib.parse.urlsplit(base)
    if parsed.scheme not in ("http", "https") or not parsed.netloc:
        raise ValueError("teacher base URL must be HTTP(S)")
    if parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise ValueError("teacher base URL cannot contain credentials, query or fragment")
    if parsed.path.endswith("/chat/completions"):
        return base
    return base + ("/chat/completions" if parsed.path.endswith("/v1") else "/v1/chat/completions")


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def error(status: int, message: str) -> dict:
    return {"status": status, "body": {"error": {"message": message, "type": "teacher_error"}}}


class TeacherForwarder:
    """The broker supplies the run identity; callers cannot select an endpoint or credential."""

    def __init__(self, settings=None, *, owner_dir: Path | None = None):
        settings = os.environ if settings is None else settings
        missing = [
            name
            for name in ("TEACHER_MODEL", "TEACHER_BASE_URL", "TEACHER_TOKEN")
            if not settings.get(name)
        ]
        if missing:
            raise ValueError("missing teacher settings: " + ", ".join(missing))
        self.model = settings["TEACHER_MODEL"]
        self._url = chat_completions_url(settings["TEACHER_BASE_URL"])
        self._key = settings["TEACHER_TOKEN"]
        self.owner_dir = owner_dir
        self._opener = urllib.request.build_opener(NoRedirect)

    def chat(self, payload) -> dict:
        if not isinstance(payload, dict) or len(json.dumps(payload).encode()) > MAX_BODY:
            return error(400, "teacher request must be a JSON object under 900 KB")
        if payload.get("model", self.model) != self.model:
            return error(403, "teacher model is pinned to " + self.model)
        if payload.get("stream"):
            return error(400, "teacher relay supports stream=false only")
        if set(payload) & {"base_url", "api_key", "url", "endpoint", "headers", "api_base"}:
            return error(400, "teacher endpoint and credentials are operator-controlled")
        messages = payload.get("messages")
        if (
            not isinstance(messages, list)
            or not messages
            or not all(
                isinstance(m, dict)
                and m.get("role") in ("system", "developer", "user", "assistant", "tool")
                for m in messages
            )
        ):
            return error(400, "teacher messages must be a nonempty chat message list")
        body = dict(payload, model=self.model, stream=False)
        # The configured Qwen endpoint supports this switch; data generation defaults
        # to direct answers while callers can explicitly request reasoning.
        if self.model.lower().startswith("qwen"):
            body.setdefault("enable_thinking", False)
        if "max_completion_tokens" not in body:
            body.setdefault("max_tokens", 4096)
        try:
            timeout = min(120, remaining_seconds(self.owner_dir))
        except ValueError:
            return error(410, "run ended; teacher access is closed")
        request = urllib.request.Request(
            self._url,
            data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json", "Authorization": "Bearer " + self._key},
            method="POST",
        )
        try:
            with self._opener.open(request, timeout=timeout) as response:
                data = response.read(MAX_REPLY + 1)
            if len(data) > MAX_REPLY:
                return error(502, "teacher response exceeds relay limit")
            result = json.loads(data)
            if len(json.dumps(result).encode()) > MAX_REPLY:
                return error(502, "teacher JSON response exceeds relay limit")
            if (
                not isinstance(result, dict)
                or not isinstance(result.get("choices"), list)
                or not result["choices"]
            ):
                return error(502, "teacher returned an invalid chat completion")
        except urllib.error.HTTPError as exc:
            # Do not echo upstream bodies: gateway error pages may contain auth/debug data.
            status = exc.code if exc.code in (400, 401, 403, 404, 408, 429) else 502
            return error(status, f"teacher upstream returned HTTP {exc.code}")
        except (TimeoutError, OSError, ValueError):
            return error(502, "teacher upstream unavailable or returned invalid JSON")
        try:
            remaining_seconds(self.owner_dir)
        except ValueError:
            return error(410, "run ended; teacher access is closed")
        return {"status": 200, "body": result}
