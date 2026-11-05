"""Assembly machinery: how named components get wired into the agent loop.

This subpackage is plumbing, not research. It answers "where do I plug things in";
what you plug in lives in `components/`. Three files:

    pipeline.py   the assembly itself — which component fills which slot
    registry.py   name -> component object, lazily and without ever raising at import
    events.py     the lifecycle hook bus

Deliberately kept import-light: nothing here imports a component at module load time.
See the note in registry.py for why that matters.
"""
