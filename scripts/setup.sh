#!/bin/bash
# setup.sh check | prepare: what the paper's replays need on this host, and (prepare) installing it. Idempotent.
#
# Written for the evaluation host: an arm64 bare-metal machine with 64 cores and 128 GB (AWS c7g.metal), Ubuntu 24.04,
# Linux 6.17, cgroup v2. prepare installs: packages; Docker CE on containerd's image store (the containerd namespace
# `moby`, where tideletd finds the task images); Go; tidelet (make build); the CNI plugins; CRIU 4.1 with the
# socket patch kernels 6.16+ need; a 64 GB swap file; uv and this checkout's venv; the Claude Code versions the
# recordings were made with; and the task images from images/ (downloaded beforehand, see the README).
#
#   scripts/setup.sh prepare      # needs sudo
#   scripts/setup.sh check        # ok / MISSING per item; exit 0 when everything is in place
#
# Environment: TIDELET (the tidelet checkout, default the parent of this TideBox), GO_VERSION (1.27.3), SWAP_GB (64),
# SWAPFILE (/swapfile), CNI_VERSION (v1.9.1), CRIU_SRC (~/src/criu).
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
TIDEBOX=$(cd "$HERE/.." && pwd)
TIDELET=${TIDELET:-$(cd "$TIDEBOX/.." && pwd)}
PY=$TIDEBOX/.venv/bin/python
GO_VERSION=${GO_VERSION:-1.27.3}
SWAP_GB=${SWAP_GB:-64}
SWAPFILE=${SWAPFILE:-/swapfile}
CNI_VERSION=${CNI_VERSION:-v1.9.1}
CRIU_SRC=${CRIU_SRC:-$HOME/src/criu}
CLAUDE_VERSIONS=$HOME/.local/share/claude/versions
IMAGE_SETS="algotune tb4 tbsci"   # images/<set>/: algotune, Terminal-Bench 4.0, Terminal-Bench Science
export PATH=$PATH:/usr/local/go/bin:$HOME/.local/bin
mode=${1:-check}
ok=0; missing=0
have() { echo "  ok       $*"; ok=$((ok + 1)); }
lack() { echo "  MISSING  $*"; missing=$((missing + 1)); }
log() { echo "$(date -u '+%H:%M:%S') $*"; }
die() { log "ERROR: $*" >&2; exit 1; }

# The Claude Code versions the recordings under data/ were made with.
claude_versions() {
  python3 -I - "$TIDEBOX/data" <<'EOF'
import json, pathlib, sys
vs = set()
for m in pathlib.Path(sys.argv[1]).glob("*/*/*/meta.json"):
    try: j = json.loads(m.read_text())
    except ValueError: continue
    v = j.get("claude_code_version") or pathlib.Path(j.get("claude_binary") or "").name
    if v: vs.add(str(v))
print(" ".join(sorted(vs)))
EOF
}

# images_present SET: how many of the set's images (images/<set>/images.txt) the image store has.
images_present() {
  local list=$TIDEBOX/images/$1/images.txt n=0 img
  [ -f "$list" ] || { echo 0; return; }
  local have_now; have_now=$(sudo docker images --format '{{.Repository}}:{{.Tag}}' 2>/dev/null)
  while read -r img; do [ -n "$img" ] && echo "$have_now" | command grep -qxF "$img" && n=$((n + 1)); done < "$list"
  echo $n
}

check() {
  echo "host:"
  sudo -n true 2>/dev/null && have "sudo without a password" || lack "sudo without a password (tideletd, the driver and docker run as root)"
  [ -f /sys/fs/cgroup/cgroup.controllers ] && have "cgroup v2" || lack "cgroup v2"
  echo "docker and containerd:"
  if sudo docker info > /dev/null 2>&1; then
    have "docker $(sudo docker version --format '{{.Server.Version}}' 2>/dev/null)"
    sudo docker info 2>/dev/null | command grep -q "driver-type: io.containerd.snapshotter" && have "containerd image store (namespace moby)" || lack "docker on the containerd image store (scripts/setup.sh prepare)"
  else lack "docker"; fi
  for b in containerd runc containerd-shim-runc-v2; do command -v "$b" > /dev/null && have "$b" || lack "$b"; done
  echo "tidelet:"
  command -v go > /dev/null && have "$(go version)" || lack "go $GO_VERSION"
  branch=$(git -C "$TIDELET" rev-parse --abbrev-ref HEAD 2>/dev/null)
  [ "$branch" = paper ] && have "tidelet on the paper branch" || lack "tidelet on the paper branch (on ${branch:-no branch}: git -C $TIDELET switch paper)"
  [ -x "$TIDELET/bin/tideletd" ] && [ -x "$TIDELET/bin/tidelet" ] && [ -x "$TIDELET/bin/tidelet-init" ] && have "$TIDELET/bin (tideletd, tidelet, tidelet-init)" || lack "tidelet binaries (cd $TIDELET && make build)"
  [ -x /opt/cni/bin/bridge ] && have "CNI plugins" || lack "CNI plugins in /opt/cni/bin"
  # tideletd writes its own network (10-tidelet.conflist, bridge tidelet0, 10.88.0.0/16) only into an empty
  # /etc/cni/net.d; with another network there it would join that one instead.
  others=$(ls /etc/cni/net.d 2>/dev/null | command grep -E '\.(conflist|conf|json)$' | command grep -vx '10-tidelet.conflist' | tr '\n' ' ')
  [ -z "$others" ] && have "no other CNI network in /etc/cni/net.d" || lack "other CNI networks in /etc/cni/net.d ($others): move them away"
  if command -v criu > /dev/null || [ -x /usr/local/sbin/criu ]; then have "criu $(sudo "$(command -v criu || echo /usr/local/sbin/criu)" --version 2>/dev/null | head -1 | cut -d' ' -f2)"
  else lack "criu (TideSand checkpoints sandboxes)"; fi
  [ -n "$(swapon --noheadings --show 2>/dev/null)" ] && have "swap $(free -g | awk '/Swap/{print $2}') GB" || lack "swap ($SWAP_GB GB)"
  echo "TideBox:"
  [ -x "$PY" ] && "$PY" -c "import tidebox, aiohttp, matplotlib, numpy" 2>/dev/null && have "$TIDEBOX/.venv" || lack "$TIDEBOX/.venv"
  echo "data (downloaded, see the README):"
  for d in algotune/run4-max-50 terminal-bench-4.0/run2-medium terminal-bench-science/run1-rand50 algotune/run1-deepseek-v4.1-flash \
           terminal-bench-4.0/run1-deepseek-v4.1-flash terminal-bench-science/run1-deepseek-v4.1-flash; do
    n=$(ls "$TIDEBOX/data/$d"/*/tape.json 2>/dev/null | wc -l)
    [ "$n" -gt 0 ] && have "data/$d ($n recordings)" || lack "data/$d"
  done
  for s in $IMAGE_SETS; do
    [ -f "$TIDEBOX/images/$s/images.txt" ] || { lack "images/$s (not downloaded)"; continue; }
    total=$(command grep -c . "$TIDEBOX/images/$s/images.txt")
    n=$(images_present "$s")
    [ "$total" -gt 0 ] && [ "$n" -ge "$total" ] && have "images/$s: $n images loaded" || lack "images/$s: $n of ${total:-?} images loaded"
  done
  echo "Claude Code (bind-mounted into the sandboxes, the version each recording was made with):"
  for v in $(claude_versions); do [ -f "$CLAUDE_VERSIONS/$v" ] && have "Claude Code $v" || lack "Claude Code $v"; done
  echo "$ok in place, $missing missing"
  [ "$missing" -eq 0 ]
}

apt_install() { sudo DEBIAN_FRONTEND=noninteractive apt-get install -y -qq "$@" > /dev/null; }

prepare() {
  sudo -n true 2>/dev/null || die "prepare needs sudo"
  log "packages"
  sudo apt-get update -qq
  apt_install build-essential git curl jq pkg-config python3 python3-venv util-linux ca-certificates gnupg \
    libprotobuf-c-dev protobuf-c-compiler protobuf-compiler python3-protobuf libnl-3-dev libnet1-dev libcap-dev libbsd-dev \
    libgnutls28-dev libnftables-dev libaio-dev uuid-dev
  if ! sudo docker info > /dev/null 2>&1; then
    log "docker CE (with containerd and runc)"
    sudo install -m 0755 -d /etc/apt/keyrings
    curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor --yes -o /etc/apt/keyrings/docker.gpg
    echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu $(. /etc/os-release && echo "$VERSION_CODENAME") stable" \
      | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
    sudo apt-get update -qq
    apt_install docker-ce docker-ce-cli containerd.io docker-buildx-plugin
  fi
  if ! sudo docker info 2>/dev/null | command grep -q "driver-type: io.containerd.snapshotter"; then
    log "docker on the containerd image store (namespace moby, where tideletd finds the images)"
    echo '{"features": {"containerd-snapshotter": true}}' | sudo tee /etc/docker/daemon.json > /dev/null
    sudo systemctl restart docker
  fi
  if ! go version 2>/dev/null | command grep -q "go$GO_VERSION"; then
    log "go $GO_VERSION"
    curl -fsSL "https://go.dev/dl/go${GO_VERSION}.linux-$(dpkg --print-architecture).tar.gz" -o /tmp/go.tgz \
      && sudo rm -rf /usr/local/go && sudo tar -C /usr/local -xzf /tmp/go.tgz && rm -f /tmp/go.tgz || die "go install failed"
  fi
  [ "$(git -C "$TIDELET" rev-parse --abbrev-ref HEAD 2>/dev/null)" = paper ] || die "$TIDELET is not on tidelet's paper branch (git -C $TIDELET switch paper)"
  log "tidelet: make build"
  (cd "$TIDELET" && make build > /dev/null) || die "make build failed in $TIDELET"
  if [ ! -x /opt/cni/bin/bridge ]; then
    log "CNI plugins $CNI_VERSION"
    tmp=$(mktemp -d) && curl -fsSL -o "$tmp/cni.tgz" "https://github.com/containernetworking/plugins/releases/download/$CNI_VERSION/cni-plugins-linux-$(dpkg --print-architecture)-$CNI_VERSION.tgz" \
      && sudo mkdir -p /opt/cni/bin && sudo tar -C /opt/cni/bin -xzf "$tmp/cni.tgz" && rm -rf "$tmp" || die "CNI plugins install failed"
  fi
  if ! command -v criu > /dev/null && [ ! -x /usr/local/sbin/criu ]; then
    log "criu 4.1 with the socket patch (scripts/criu-sockets.patch), from source"
    mkdir -p "$(dirname "$CRIU_SRC")"
    [ -d "$CRIU_SRC" ] || git clone -q --depth 1 --branch v4.1 https://github.com/checkpoint-restore/criu.git "$CRIU_SRC"
    [ -f "$CRIU_SRC/.sockets-patched" ] || { (cd "$CRIU_SRC" && git apply "$HERE/criu-sockets.patch") && touch "$CRIU_SRC/.sockets-patched"; } || die "the CRIU patch did not apply"
    (cd "$CRIU_SRC" && make -j"$(nproc)" > ../criu-build.log 2>&1 && sudo make install-criu >> ../criu-build.log 2>&1) || die "the CRIU build failed; see $CRIU_SRC/../criu-build.log"
  fi
  if [ -z "$(swapon --noheadings --show 2>/dev/null)" ]; then
    log "swap: $SWAP_GB GB at $SWAPFILE"
    [ -f "$SWAPFILE" ] || { sudo fallocate -l "${SWAP_GB}G" "$SWAPFILE" && sudo chmod 600 "$SWAPFILE" && sudo mkswap "$SWAPFILE" > /dev/null; }
    sudo swapon "$SWAPFILE" || die "swapon failed"
  fi
  command -v uv > /dev/null || { log "uv"; curl -LsSf https://astral.sh/uv/install.sh | sh > /dev/null; }
  if ! "$PY" -c "import tidebox, aiohttp, matplotlib, numpy" 2>/dev/null; then
    log "TideBox venv"
    (cd "$TIDEBOX" && uv venv -q .venv --python 3.12 && uv pip install -q --python .venv/bin/python -e '.[workload,dev]') || die "the venv could not be built"
  fi
  installed=0
  for v in $(claude_versions); do
    [ -f "$CLAUDE_VERSIONS/$v" ] && continue
    log "Claude Code $v (the version recordings were made with)"
    curl -fsSL https://claude.ai/install.sh | bash -s "$v" > /dev/null 2>&1 && installed=1 || log "  WARNING: Claude Code $v could not be installed"
  done
  # Installing a version points ~/.local/bin/claude at it: point it back at the newest installed.
  newest=$(ls "$CLAUDE_VERSIONS" 2>/dev/null | sort -V | tail -1)
  [ "$installed" = 1 ] && [ -n "$newest" ] && ln -sfn "$CLAUDE_VERSIONS/$newest" "$HOME/.local/bin/claude"
  for s in $IMAGE_SETS; do
    d=$TIDEBOX/images/$s
    [ -f "$d/images.txt" ] || { log "images/$s not downloaded; skipped (see the README)"; continue; }
    total=$(command grep -c . "$d/images.txt"); n=$(images_present "$s")
    [ "$n" -ge "$total" ] && { log "images/$s: all $total images loaded"; continue; }
    log "images/$s: checking and loading $total images (needs about as much free disk again as the archive)"
    (cd "$d" && sha256sum -c --quiet ./*.sha256) || die "images/$s does not match its checksums"
    cat "$d"/*.tar.gz.part* | sudo docker load > "$d/load.log" 2>&1 || die "docker load failed; see $d/load.log"
  done
  log "prepare done"
  check
}

case $mode in
  check) check;;
  prepare) prepare;;
  *) die "usage: setup.sh check | prepare";;
esac
