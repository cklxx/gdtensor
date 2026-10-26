#!/usr/bin/env bash
# Validate the staged tree on the configured SSH host. Never run tests locally.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

HOST="${GDTENSOR_REMOTE_HOST:-digest}"
if [ -n "${GDTENSOR_REMOTE_USER:-}" ]; then
  TARGET="${GDTENSOR_REMOTE_USER}@${HOST}"
else
  TARGET="$HOST"
fi
case "$TARGET" in
  ""|-*) printf '%s\n' '[remote-gate] invalid SSH target' >&2; exit 2 ;;
esac
RROOT="${GDTENSOR_REMOTE_ROOT:-gdtensor-verify}"
PROXY="${GDTENSOR_REMOTE_PROXY:-}"
MODE="${1:-precommit}"
case "$MODE" in
  precommit) SCRIPT="scripts/precommit.sh" ;;
  full) SCRIPT="scripts/full_check.sh" ;;
  *) printf '[remote-gate] unknown mode: %s\n' "$MODE" >&2; exit 2 ;;
esac

TREE="$(git write-tree)"
LOGDIR="log/fix-df1/remote-$MODE"
mkdir -p "$LOGDIR"
LOGFILE="$LOGDIR/$TREE-$(date +%s)-$$.log"
printf '[remote-gate:%s] tree=%s host=%s\n' "$MODE" "$TREE" "$TARGET"

# Pass parameters as shell-quoted arguments, not interpolated script fragments.
# Tar consumes SSH stdin; bash receives its program through the -c argument.
REMOTE='
set -euo pipefail
TREE="$1"; RROOT="$2"; SCRIPT="$3"; PROXY="$4"
case "$RROOT" in
  /*) ROOT="$RROOT" ;;
  *) ROOT="$HOME/$RROOT" ;;
esac
mkdir -p "$ROOT/precommit/$TREE"
exec 9>"$ROOT/precommit/.lock-$TREE"
flock 9
DEST="$(mktemp -d "$ROOT/precommit/$TREE/run-XXXXXXXX")"
printf "[remote-gate] host=%s snapshot=%s\n" "$(hostname)" "$DEST"
tar -xf - -C "$DEST"
cd "$DEST"
git init -q
git -c core.autocrlf=false add -A
GOT="$(git write-tree)"
[ "$GOT" = "$TREE" ] || {
  printf "[remote-gate] tree mismatch: %s != %s\n" "$GOT" "$TREE" >&2
  exit 3
}
# Each snapshot owns its environment. uv shares only its package cache.
export UV_PROJECT_ENVIRONMENT="$DEST/.venv"
export PATH="$HOME/.local/bin:$PATH"
unset PYTHONPATH VIRTUAL_ENV
export PYTHONPATH="."
# Wheels are 10-50 MiB; the remote uplink is slow. Give uv a real download budget
# instead of failing on the 30s default (observed: numpy wheel timeout).
export UV_HTTP_TIMEOUT=300
if [ -n "$PROXY" ]; then
  export HTTPS_PROXY="$PROXY" HTTP_PROXY="$PROXY"
  export https_proxy="$PROXY" http_proxy="$PROXY"
fi
uv --version
GDTENSOR_REMOTE_TEST=1 bash "$SCRIPT"
'
printf -v REMOTE_CMD 'bash -c %q -- %q %q %q %q' "$REMOTE" "$TREE" "$RROOT" "$SCRIPT" "$PROXY"

set +e
git archive --format=tar "$TREE" \
  | ssh -o BatchMode=yes -o ConnectTimeout=15 "$TARGET" "$REMOTE_CMD" \
  2>&1 | tee "$LOGFILE"
rcs=("${PIPESTATUS[@]}")
set -e
status=0
for code in "${rcs[@]}"; do
  if [ "$code" -ne 0 ]; then status="$code"; break; fi
done
if [ "$status" -ne 0 ]; then
  printf '[remote-gate] FAILED exit=%s; transcript=%s\n' "$status" "$LOGFILE" >&2
  exit "$status"
fi
# A changing index must not reuse evidence for a different commit tree.
[ "$(git write-tree)" = "$TREE" ] || {
  printf '%s\n' '[remote-gate] staged tree changed during checks; commit blocked' >&2
  exit 4
}
printf '[remote-gate] OK tree=%s; transcript=%s\n' "$TREE" "$LOGFILE"
