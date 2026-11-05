"""Toolset installer: builds a single self-contained shell command that materialises the tool
engine and its PATH shims inside the remote sandbox.

This package IS part of the harness — the tool surface the agent gets is yours to design
(add/remove verbs in shims.VERBS, change their semantics in engine.py, or drop the whole
discrete-verb approach and hand the model raw bash). See EDITABLE.md.

`install_command()` returns ONE bash line the agent loop runs once per episode (before the model
takes any step). It carries the whole payload as base64-encoded python source, so the only shell
metacharacters in the returned string are the fixed `python -c "..."` wrapper — file contents
(quotes, newlines, `$`, backslashes) are all sealed inside the base64 blob and never touch the
shell. On success it prints TOOLS_INSTALL_OK.
"""
import base64
from pathlib import Path

from minisweagent.tools import shims

# Runs INSIDE the sandbox. `__FILES__` is replaced with a repr() of {path: content}.
_INSTALLER = r'''
import os
files = __FILES__
for path, content in files.items():
    d = os.path.dirname(path)
    if d and not os.path.isdir(d):
        os.makedirs(d, exist_ok=True)
    with open(path, "w") as f:
        f.write(content)
for path in files:
    if path.startswith("/usr/local/bin/"):
        try:
            os.chmod(path, 0o755)
        except Exception:
            pass
print("TOOLS_INSTALL_OK")
'''


def _file_map() -> dict:
    engine_src = (Path(__file__).parent / "engine.py").read_text()
    files = {"/root/.tools/engine.py": engine_src}
    for verb in shims.VERBS:
        files["/usr/local/bin/" + verb] = shims.shim_body(verb)
    files["/usr/local/bin/submit"] = shims.SUBMIT_SHIM
    return files


def install_command() -> str:
    installer = _INSTALLER.replace("__FILES__", repr(_file_map()))
    blob = base64.b64encode(installer.encode("utf-8")).decode("ascii")
    payload = "import base64;exec(base64.b64decode('{}').decode())".format(blob)
    return 'python3 -c "{p}" || python -c "{p}"'.format(p=payload)
