#!/usr/bin/env python3
"""Tool engine — discrete file navigation/edit tools for the SWE weak baseline.

This is a pre-bash-unification, SWE-agent-style tool surface: the model drives a small set of
discrete verbs (open/scroll_down/create/edit/insert/submit) instead of one free-form
shell command. For a weak base model this is DELIBERATELY harder than raw bash — each verb is
another syntax it can get wrong — which is exactly the headroom the coevolution researcher is
meant to close (make these tools stronger, or teach/allow raw bash again).

The surface is deliberately thin. There is no search of any kind — not across the tree, not
within the open file — no way to jump to a line, no scroll back up, and no find-by-name;
navigation is open-then-scroll-forward and nothing else. Editing applies whatever you give
it, including code that does not parse. Widening any of that means writing the verb — the
implementations are not hiding somewhere disabled.

Installed by the harness into the sandbox at /root/.tools/engine.py and dispatched through thin sh
shims under /usr/local/bin (see shims.py). stdlib-only; runs on any python3.6+. State (the
currently-open file and window position) persists across turns in /root/.tools/state.json, because
each turn is a fresh, stateless subshell.

Placement note: this file and the shims live outside /testbed, so they never appear in the graded
git diff.
"""
import fnmatch
import os
import sys

STATE_DIR = "/root/.tools"
STATE_FILE = os.path.join(STATE_DIR, "state.json")
ROOT = "/testbed"
WINDOW = 100          # lines shown per view
SEARCH_CAP = 100      # max matches reported
MAX_LINE = 500        # per-line display truncation


# --------------------------------------------------------------------------- state
def _load():
    try:
        import json
        with open(STATE_FILE) as f:
            return json.load(f)
    except Exception:
        return {"path": None, "first": 1}


def _save(st):
    try:
        import json
        os.makedirs(STATE_DIR, exist_ok=True)
        with open(STATE_FILE, "w") as f:
            json.dump(st, f)
    except Exception:
        pass


# --------------------------------------------------------------------------- io helpers
def _abspath(p):
    if not p:
        return p
    if not os.path.isabs(p):
        p = os.path.join(ROOT, p)
    return os.path.normpath(p)


def _read_text(path):
    with open(path, "r", errors="replace") as f:
        return f.read()


def _write_text(path, text):
    with open(path, "w", errors="replace") as f:
        f.write(text)


def _get_lines(path):
    """Logical lines, without the trailing empty produced by a final newline."""
    lines = _read_text(path).split("\n")
    if lines and lines[-1] == "":
        lines = lines[:-1]
    return lines


def _put_lines(path, lines):
    _write_text(path, "\n".join(lines) + "\n")


def _stdin_body():
    """Heredoc body for edit/insert/create, or None if no body was piped in.

    Strips exactly one trailing newline (the heredoc's own terminator artifact)."""
    try:
        if sys.stdin is None or sys.stdin.isatty():
            return None
        data = sys.stdin.read()
    except Exception:
        return None
    if data == "":
        return None
    if data.endswith("\n"):
        data = data[:-1]
    return data


def _lint_ok_or_revert(path, original):
    """WEAKENED: the syntax-check safety net is removed. edit/insert are applied verbatim,
    even when they introduce a Python SyntaxError — nothing is reverted. Always returns None."""
    return None


# --------------------------------------------------------------------------- rendering
def _render(path, first):
    if not path or not os.path.isfile(path):
        return "[No file open. Use: open <path>]"
    lines = _get_lines(path)
    total = len(lines)
    if total == 0:
        return "[File: {} (0 lines total)]".format(path)
    first = max(1, min(first, total))
    last = min(total, first + WINDOW - 1)
    out = ["[File: {} ({} lines total)]".format(path, total)]
    if first > 1:
        out.append("({} more lines above)".format(first - 1))
    width = len(str(last))
    for i in range(first, last + 1):
        content = lines[i - 1]
        if len(content) > MAX_LINE:
            content = content[:MAX_LINE] + " ...[truncated]"
        out.append("{}:{}".format(str(i).rjust(width), content))
    if last < total:
        out.append("({} more lines below)".format(total - last))
    return "\n".join(out)


# --------------------------------------------------------------------------- verbs
def cmd_open(argv):
    """Open a file at the top. There is no line argument — that was a jump-to-line under
    another name, and the docstring above has always said this surface does not have one.
    Reaching line 1740 costs seventeen scroll_down calls, and there is no scroll back."""
    if not argv:
        return "Usage: open <path>"
    path = _abspath(argv[0])
    if not os.path.isfile(path):
        return "open: no such file: {}".format(path)
    if len(argv) >= 2:
        return ("open: takes a path only — this toolset has no jump-to-line. "
                "Use scroll_down to move through the file.")
    _save({"path": path, "first": 1})
    return _render(path, 1)


def cmd_scroll_down():
    st = _load()
    path = st.get("path")
    if not path:
        return "No file open. Use: open <path>"
    first = max(1, int(st.get("first", 1)))
    total = len(_get_lines(path))
    if first + WINDOW > total:
        # At the end of the file. `first` used to keep growing past `total` while `_render`
        # clamped it back, so every further scroll returned the same last window with nothing
        # to say it was the same — a reader with no scroll-back and no loop detection can sit
        # here until the step budget runs out. Saying so is not a new capability; the position
        # was always knowable, it just was not reported.
        _save({"path": path, "first": first})
        return ("[End of file: {} has {} lines and you are already at the bottom. "
                "scroll_down will not move, and there is no way back up: re-open the file "
                "to start again from line 1.]\n").format(path, total) \
            + _render(path, first)
    st["first"] = first + WINDOW
    _save(st)
    return _render(path, st["first"])


def cmd_create(argv):
    if not argv:
        return "Usage: create <path> [ <<'EOF' ... EOF ]"
    path = _abspath(argv[0])
    if os.path.isfile(path):
        return "create: file already exists: {} (use: open {} then edit)".format(path, argv[0])
    d = os.path.dirname(path)
    if d and not os.path.isdir(d):
        os.makedirs(d, exist_ok=True)
    body = _stdin_body()
    if body is None:
        _write_text(path, "")
    else:
        _put_lines(path, body.split("\n"))
    _save({"path": path, "first": 1})
    note = ("[Created {} — NOTE: brand-new files are untracked and are NOT graded. "
            "Fix the bug by editing existing source; use created files only for scratch/repro.]\n")
    return note.format(path) + _render(path, 1)


def cmd_edit(argv):
    st = _load()
    path = st.get("path")
    if not path:
        return "No file open. Use: open <path>"
    if len(argv) < 2:
        return "Usage: edit <start_line> <end_line> <<'EOF'\\n<new code>\\nEOF"
    try:
        start = int(argv[0])
        end = int(argv[1])
    except ValueError:
        return "edit: start and end must be integers"
    body = _stdin_body()
    if body is None:
        return ("edit: no replacement text received. Supply it with a heredoc, e.g.\n"
                "edit {} {} <<'EOF'\n    <new code>\nEOF".format(start, end))
    new_lines = body.split("\n")
    lines = _get_lines(path)
    total = len(lines)
    if start < 1 or end < start or start > total + 1:
        return "edit: invalid range {}..{} (file has {} lines)".format(start, end, total)
    end = min(end, total)
    original = _read_text(path)
    _put_lines(path, lines[:start - 1] + new_lines + lines[end:])
    err = _lint_ok_or_revert(path, original)
    if err:
        return "[Edit rejected: it would introduce a syntax error — file left unchanged]\n" + err
    st["first"] = max(1, start - 3)
    _save(st)
    return "[File updated. Showing the region around your edit.]\n" + _render(path, st["first"])


def cmd_insert(argv):
    st = _load()
    path = st.get("path")
    if not path:
        return "No file open. Use: open <path>"
    if len(argv) < 1:
        return "Usage: insert <line> <<'EOF'\\n<new code>\\nEOF   (0 = top of file)"
    try:
        at = int(argv[0])
    except ValueError:
        return "insert: line must be an integer"
    body = _stdin_body()
    if body is None:
        return ("insert: no text received. Supply it with a heredoc, e.g.\n"
                "insert {} <<'EOF'\n    <new code>\nEOF".format(at))
    new_lines = body.split("\n")
    lines = _get_lines(path)
    total = len(lines)
    if at < 0 or at > total:
        return "insert: line {} out of range (0..{})".format(at, total)
    original = _read_text(path)
    _put_lines(path, lines[:at] + new_lines + lines[at:])
    err = _lint_ok_or_revert(path, original)
    if err:
        return "[Insert rejected: it would introduce a syntax error — file left unchanged]\n" + err
    st["first"] = max(1, at - 2)
    _save(st)
    return "[File updated after line {}.]\n".format(at) + _render(path, st["first"])


_DISPATCH = {
    "open": lambda r: cmd_open(r),
    "scroll_down": lambda r: cmd_scroll_down(),
    "create": lambda r: cmd_create(r),
    "edit": lambda r: cmd_edit(r),
    "insert": lambda r: cmd_insert(r),
}


def main():
    argv = sys.argv[1:]
    if not argv:
        print("tool: no verb given")
        return
    verb, rest = argv[0], argv[1:]
    fn = _DISPATCH.get(verb)
    if fn is None:
        print("tool: unknown verb '{}'".format(verb))
        return
    try:
        print(fn(rest))
    except Exception as e:  # never crash the turn; a normal observation is better
        print("tool error in {}: {}".format(verb, e))


if __name__ == "__main__":
    main()
