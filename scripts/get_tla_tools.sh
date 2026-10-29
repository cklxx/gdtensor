#!/usr/bin/env bash
# Reproducibly provision the pinned tla2tools.jar for formal/tla/run_tlc.sh.
#
# The TLA+ tools are NOT a pip/uv dependency, so their version must be pinned
# here instead of in pyproject.toml: an official GitHub release asset plus its
# sha256. The jar is read-only input to the TLA owner's run_tlc.sh, which
# consumes it through the TLA2TOOLS_JAR environment variable.
#
# Behaviour:
#   * The pinned jar is cached under log/tools/ (git-ignored) and reused when it
#     already matches the pin, so a second run is fully offline.
#   * Any cached or downloaded jar is sha256-verified against the pin; a mismatch
#     aborts non-zero (never proceeds with an unverified jar).
#   * The jar is smoke-tested by launching tlc2.TLC itself (not just a shell
#     presence check).
#   * The resolved jar ABSOLUTE path is printed on stdout for callers:
#       TLA2TOOLS_JAR="$(scripts/get_tla_tools.sh)" bash formal/tla/run_tlc.sh
#
# Pin provenance: official tlaplus/tlaplus GitHub release v1.8.0 asset
#   tla2tools.jar: asset 683491413, created 2026-10-28T01:50:09Z.
# The upstream v1.8.0 asset changed in place. This SHA matches the
# official GitHub release asset digest, verified on 2026-10-29.
# Re-pin only after verifying the official asset and its digest.
set -euo pipefail

TLA2TOOLS_VERSION="v1.8.0"
TLA2TOOLS_URL="https://github.com/tlaplus/tlaplus/releases/download/${TLA2TOOLS_VERSION}/tla2tools.jar"
TLA2TOOLS_SHA256="fc6878c603f0fdc18b9fe3e5bf51860cb90ab5b1154e3c371660c036d6bdd63e"

HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"
CACHE_DIR="$ROOT/log/tools"
JAR="$CACHE_DIR/tla2tools-${TLA2TOOLS_VERSION}-${TLA2TOOLS_SHA256:0:12}.jar"

if ! command -v java >/dev/null 2>&1; then
  echo "ERROR: a JRE/JDK (java) is required to run the pinned TLC jar" >&2
  exit 1
fi

_sha256() {
  if command -v shasum >/dev/null 2>&1; then
    shasum -a 256 "$1" | awk '{print $1}'
  else
    sha256sum "$1" | awk '{print $1}'
  fi
}

verify() {  # $1 = jar path
  local got
  got="$(_sha256 "$1")"
  if [ "$got" != "$TLA2TOOLS_SHA256" ]; then
    echo "ERROR: tla2tools jar sha256 does not match the pinned release" >&2
    echo "  pin:   $TLA2TOOLS_VERSION  $TLA2TOOLS_SHA256" >&2
    echo "  got:   $got  ($1)" >&2
    exit 1
  fi
}

# Caller may point TLA2TOOLS_JAR at an already-provisioned jar (e.g. on a host
# without GitHub egress). We accept it only if its sha256 matches the pin, then
# echo the absolute path; no download is attempted.
if [ -n "${TLA2TOOLS_JAR:-}" ] && [ -f "$TLA2TOOLS_JAR" ]; then
  verify "$TLA2TOOLS_JAR"
  echo "$TLA2TOOLS_JAR"
  exit 0
fi

if [ ! -f "$JAR" ]; then
  mkdir -p "$CACHE_DIR"
  tmp="$JAR.part"
  echo "[get_tla_tools] downloading $TLA2TOOLS_VERSION from $TLA2TOOLS_URL" >&2
  curl -fsSL --retry 3 --retry-connrefused -o "$tmp" "$TLA2TOOLS_URL"
  verify "$tmp"
  mv "$tmp" "$JAR"
else
  verify "$JAR"
fi

# Launch TLC itself to prove the pinned jar's main class loads (not merely a
# file of the right size/hash). `tlc2.TLC` with no model prints its version
# banner then exits non-zero, so capture the banner and match it rather than the
# exit code (and `|| true` under pipefail).
banner="$(java -cp "$JAR" tlc2.TLC 2>&1 || true)"
case "$banner" in
  *"TLC2 Version"*) ;;
  *)
    echo "ERROR: pinned tla2tools jar failed to launch tlc2.TLC" >&2
    exit 1
    ;;
esac

echo "$JAR"
