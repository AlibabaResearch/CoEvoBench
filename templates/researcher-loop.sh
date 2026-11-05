#!/bin/sh
# Frozen researcher driver: budgeted continuation, recovery and submission checks.
exec python3 -m coevobench.research.loop "$@"
