"""Sandbox-side dispatch shims for the toolset.

Each verb becomes a tiny POSIX-sh script on PATH (/usr/local/bin/<verb>) that execs the tool
engine at /root/.tools/engine.py. `exec` hands the shim's argv AND stdin straight to python, so
heredoc bodies used by edit/insert/create flow through unchanged. python3 is preferred, with a
python fallback for images that only ship the unversioned name.

`submit` is special: it is NOT an engine verb but a pure sentinel echo, kept here so the model has
a single uniform "verb" vocabulary. The platform's _check_finished treats the sentinel on stdout
with rc==0 as the finish signal.

VERBS below is what this harness ADDS to the sandbox — editing this list is a first-class harness
change (see EDITABLE.md), but adding a name here only helps if `engine.py` implements the verb.

Note what these shims are and are not. Installing a script at /usr/local/bin/<verb> adds a name to
PATH; it does not take anything away. The sandbox is an ordinary Linux container and everything
already in it — grep, find, cat, sed, python, git, ls — remains callable. The action the harness
sends to the environment is a command string, and nothing here filters it.
"""

# The verbs this harness installs. Navigation through them is open-then-scroll-forward only: no
# search at all (not even within the open file), no jump-to-line, no scroll back, no find-by-name.
# Those verbs are not implemented in the engine — that is a real gap in the toolset, not a
# restriction on the shell.
VERBS = [
    "open", "scroll_down",
    "create", "edit", "insert",
]

SUBMIT_SENTINEL = "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"


def shim_body(verb: str) -> str:
    return (
        "#!/bin/sh\n"
        'if command -v python3 >/dev/null 2>&1; then P=python3; else P=python; fi\n'
        'exec "$P" /root/.tools/engine.py {verb} "$@"\n'
    ).format(verb=verb)


SUBMIT_SHIM = "#!/bin/sh\necho {sentinel}\n".format(sentinel=SUBMIT_SENTINEL)
