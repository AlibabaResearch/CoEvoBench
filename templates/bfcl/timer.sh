#!/bin/sh
# Budget query; the supervisor enforces the deadline.
STATE="${CEB_LIFECYCLE:-/dev/null}"
case "$1" in
  status)
    if [ -f "$STATE" ]; then
      deadline=$(python3 -c "import json;print(json.load(open('$STATE'))['deadline_at'])" 2>/dev/null)
      if [ -n "$deadline" ]; then
        left=$(python3 -c "import time;print(max(0.0,($deadline-time.time())/3600.0))" 2>/dev/null)
        echo "budget left: ${left}h"
        exit 0
      fi
    fi
    echo "budget left: unknown (no lifecycle state)"
    ;;
  *)
    echo "usage: ./timer.sh status"
    ;;
esac
