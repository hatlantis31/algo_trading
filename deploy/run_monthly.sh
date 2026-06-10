#!/usr/bin/env bash
# run_monthly.sh — cron entrypoint for the monthly IBKR rebalance.
#
# Guards so it only trades on the LAST TRADING DAY of the month:
#   1. Skip weekends (Sat/Sun).
#   2. Skip if there is another weekday later this month (i.e. only run on the
#      final Mon-Fri of the month). This makes the 28-31 cron range exact.
#
# Edit REPO_DIR and the port/flags for your setup.

set -euo pipefail

REPO_DIR="/home/ubuntu/algo_trading"
PORT=4002          # 4002 = paper Gateway, 4001 = live Gateway
EXECUTE="--execute"  # remove this to keep cron in permanent dry-run mode

cd "$REPO_DIR"

# ── Guard 1: skip weekends ────────────────────────────────────────────────
DOW=$(date +%u)          # 1=Mon … 7=Sun
if [ "$DOW" -ge 6 ]; then
    echo "$(date)  Weekend — skipping."
    exit 0
fi

# ── Guard 2: only the LAST weekday of the month ───────────────────────────
# If adding 7 days stays in the same month, this isn't the last week → but we
# still need the last *weekday*. Check: is there any later weekday this month?
TODAY=$(date +%d)
MONTH=$(date +%m)
for d in 1 2 3 4 5 6 7; do
    FUTURE=$(date -d "+$d day" +%m)
    FUTURE_DOW=$(date -d "+$d day" +%u)
    if [ "$FUTURE" = "$MONTH" ] && [ "$FUTURE_DOW" -le 5 ]; then
        echo "$(date)  Not the last trading day of the month — skipping."
        exit 0
    fi
done

echo "$(date)  Last trading day — running rebalance."

# ── Run the rebalance ─────────────────────────────────────────────────────
# Uses the repo's venv if present, else system python.
PYTHON="python3"
[ -x "$REPO_DIR/.venv/bin/python" ] && PYTHON="$REPO_DIR/.venv/bin/python"

"$PYTHON" scripts/run_live_rebalance.py --port "$PORT" $EXECUTE

echo "$(date)  Rebalance complete."
