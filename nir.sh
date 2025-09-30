#!/bin/bash

set -eu

# Idempotent Docker installer for Debian/Ubuntu
# - Optional pin: DOCKER_VERSION=28.3.3 (or 28.3)
# - Channels: CHANNEL=stable|test|nightly (default: stable)
# - Dry run: --dry-run
# - No use of OS "VERSION" from /etc/os-release

CHANNEL="${CHANNEL:-stable}"
DOWNLOAD_URL="${DOWNLOAD_URL:-https://download.docker.com}"
DOCKER_VERSION="${DOCKER_VERSION:-}"
DRYRUN=0
[ "${1:-}" = "--dry-run" ] && DRYRUN=1

run() {
  if [ "$DRYRUN" -eq 1 ]; then echo "+ $*"; else sh -c "$*"; fi
}

need_sudo() {
  if [ "$(id -u)" -ne 0 ]; then
    if command -v sudo >/dev/null 2>&1; then echo "sudo -E"; else echo ""; fi
  else
    echo ""
  fi
}

SUDO="$(need_sudo)"

# Detect distro/codename
. /etc/os-release
DISTRO="$(echo "$ID" | tr '[:upper:]' '[:lower:]')"
CODENAME="${VERSION_CODENAME:-$(lsb_release -cs 2>/dev/null || true)}"
[ -n "$DISTRO" ] || { echo "Cannot detect distro ID"; exit 1; }
[ -n "$CODENAME" ] || { echo "Cannot detect distro codename"; exit 1; }
case "$DISTRO" in
  ubuntu|debian|raspbian) : ;;
  *) echo "Unsupported distro: $DISTRO"; exit 1 ;;
esac

ARCH="$(dpkg --print-architecture)"
REPO_FILE="/etc/apt/sources.list.d/docker.list"
KEY_FILE="/etc/apt/keyrings/docker.asc"

# Ensure repo/key exist (idempotent)
if [ ! -f "$KEY_FILE" ]; then
  run "$SUDO install -m 0755 -d /etc/apt/keyrings"
  run "curl -fsSL '$DOWNLOAD_URL/linux/$DISTRO/gpg' | $SUDO tee '$KEY_FILE' >/dev/null"
  run "$SUDO chmod a+r '$KEY_FILE'"
fi

NEED_REPO_LINE="deb [arch=$ARCH signed-by=$KEY_FILE] $DOWNLOAD_URL/linux/$DISTRO $CODENAME $CHANNEL"
ADD_REPO=1
if [ -f "$REPO_FILE" ]; then
  if grep -Fq "$NEED_REPO_LINE" "$REPO_FILE"; then ADD_REPO=0; fi
fi
if [ $ADD_REPO -eq 1 ]; then
  run "echo '$NEED_REPO_LINE' | $SUDO tee '$REPO_FILE' >/dev/null"
fi

# Update package lists
run "$SUDO apt-get update -qq"

# Determine target versions (optional pin)
PKG_PIN=""
CLI_PIN=""
if [ -n "$DOCKER_VERSION" ]; then
  # Accept "28.3" or "28.3.3"
  case "$DOCKER_VERSION" in
    *[!0-9.-]*) echo "Invalid DOCKER_VERSION: $DOCKER_VERSION"; exit 1 ;;
  esac
  PATTERN="$(echo "$DOCKER_VERSION" | sed 's/-/./g').*-0~$DISTRO"
  # Resolve exact versions via apt-cache madison
  PKG_VER="$($SUDO sh -c "apt-cache madison docker-ce | awk '{print \$3}' | grep -E '^$PATTERN' | head -1")"
  [ -n "$PKG_VER" ] || { echo "Requested DOCKER_VERSION not found: $DOCKER_VERSION"; exit 1; }
  PKG_PIN="=$PKG_VER"

  CLI_VER="$($SUDO sh -c "apt-cache madison docker-ce-cli | awk '{print \$3}' | grep -E '^$PATTERN' | head -1")"
  [ -n "$CLI_VER" ] && CLI_PIN="=$CLI_VER"
fi

# Check current installed version
current_docker=""
if command -v docker >/dev/null 2>&1; then
  # format: Docker version 28.3.3, build ...
  current_docker="$(docker --version 2>/dev/null | awk '{print $3}' | tr -d ',')"
fi

# Extract desired engine version from pin, else mark as "latest"
desired_engine="latest"
if [ -n "$PKG_PIN" ]; then
  desired_engine="$(echo "$PKG_PIN" | sed 's/^=//' | cut -d'-' -f2 | cut -d'~' -f1)"
fi

# If already at the desired version (or latest without pin), skip installs
if [ -n "$current_docker" ]; then
  if [ "$desired_engine" = "latest" ]; then
    echo "Docker already installed ($current_docker). Leaving as-is."
  else
    if [ "$current_docker" = "$desired_engine" ]; then
      echo "Docker already at requested version ($current_docker). Nothing to do."
      exit 0
    else
      echo "Docker $current_docker installed; will change to $desired_engine."
    fi
  fi
fi

# Install/upgrade (CLI first if pinned)
if [ -n "$CLI_PIN" ]; then
  run "$SUDO apt-get install -y -qq --no-install-recommends docker-ce-cli$CLI_PIN"
else
  run "$SUDO apt-get install -y -qq --no-install-recommends docker-ce-cli"
fi

run "$SUDO apt-get install -y -qq --no-install-recommends \
  docker-ce$PKG_PIN containerd.io docker-buildx-plugin docker-compose-plugin docker-ce-rootless-extras"

# Enable/start docker if systemd is available
if command -v systemctl >/dev/null 2>&1; then
  run "$SUDO systemctl enable --now docker || true"
fi

# Final info
if command -v docker >/dev/null 2>&1; then
  docker --version || true
  docker compose version || true
fi

echo "Done."