#!/bin/bash
# Replay a benchmark's recordings in LLM mode under several tideletd policies, one after the
# other, the driver offering one sandbox per INTERVAL and the policy deciding what runs. For each
# policy: start tideletd with that policy and the LLM proxy, confined to a pool cgroup,
# materialise the bridge with a throwaway sandbox, run replay.py through the proxy with the
# policy's repeat, verify the results, stop tideletd. A policy whose output directory carries a
# .complete marker is skipped, so the script resumes.
#
#   DATA=data/algotune/run4-max-50 workload/replay_policies.sh strict overcommit:2 tidesand:2
#   DATA=data/terminal-bench-4.0/run2-medium INTERVAL=3 SINGLE_CONTAINER=1 workload/replay_policies.sh overcommit:8
#
# scripts/run.sh runs one experiment of the paper through this script with the experiment's settings.
#
# Arguments are policy[:repeat] (repeat 1 when omitted): `--repeat` copies of every recorded task
# are offered, one every INTERVAL seconds in a seeded random order, so the pool carries that
# multiple of the recorded demand; the policy starts, queues or rejects each request, and a
# rejected sandbox is offered again at the next interval (replay.py --interval). Output goes to
# $RUN/<policy>$OUT_SUFFIX (default RUN: output/replay/<bench>/<the recordings' directory>, the
# benchmark read off the DATA path), with tideletd.log, stdout.log, metrics.txt and run.json (the
# setup) beside the per-sandbox directories.
#
# The host side (tideletd, the driver, the controllers, containerd and Docker) is pinned to
# DRIVER_CPUS, and the pool to POOL_CPUSET with POOL_MEMORY, so the pool's figures are its own.
#
# Environment: DATA (the recordings, required), RUN, INTERVAL (seconds between requests to start
# a sandbox, default 15; the Terminal-Bench runs used 3), TIDELET (the tidelet checkout, default
# the parent of this TideBox), SEED (submission order, default 1792198064), POOL_CPUSET (0-55),
# POOL_MEMORY (112Gi), DRIVER_CPUS (56-63), SAMPLE (1s), MAX_DEFER (30m: a holding policy keeps a
# response until its execution fits, and Claude Code's timeouts are raised to an hour to match),
# NAMESPACE (containerd namespace holding the images, default moby: Docker's own image store),
# PORT_BASE (20000), ADMISSION_TIMEOUT (43200 s), POLICY_OPTS (extra --policy-opt key=value pairs
# for every policy), TIDELETD_ARGS (extra tideletd flags, e.g. --container-checkpoints), OUT_SUFFIX or OUT_NAME,
# TASKS (task names to replay instead of every recorded one, for a pilot), SINGLE_CONTAINER (1: only
# the tasks without service containers), WARMUP_IMAGE (default the first task's image), DRIVER_ARGS
# (extra replay.py flags, e.g. --speed 0), CUTOFF (seconds; unset: none. For a pilot of fixed
# length: when it elapses, every sandbox is removed, the driver records what each got done, and
# the policy counts as not complete).
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
TIDEBOX=$(cd "$HERE/.." && pwd)
TIDELET=${TIDELET:-$(cd "$TIDEBOX/.." && pwd)}
DATA=$(cd "${DATA:?set DATA to the recordings directory (data/<bench>/<run>)}" && pwd)
BENCH=${BENCH:-$(basename "$(dirname "$DATA")")}
RUN=${RUN:-$TIDEBOX/output/replay/$BENCH/$(basename "$DATA")}
POLICIES=${*:-${POLICIES:-strict}}
SEED=${SEED:-1792198064}
INTERVAL=${INTERVAL:-15}
POOL_CPUSET=${POOL_CPUSET:-0-55}
POOL_MEMORY=${POOL_MEMORY:-112Gi}
DRIVER_CPUS=${DRIVER_CPUS:-56-63}
SAMPLE=${SAMPLE:-1s}
MAX_DEFER=${MAX_DEFER:-30m}
NAMESPACE=${NAMESPACE:-moby}
PORT_BASE=${PORT_BASE:-20000}
ADMISSION_TIMEOUT=${ADMISSION_TIMEOUT:-43200}
TASKS=${TASKS:-}
DRIVER=$TIDEBOX/workload/replay.py
PROXY=http://10.88.0.1:18800
mkdir -p "$RUN"
LOG=$RUN/policies.log
log() { echo "$(date -u '+%Y-%m-%d %H:%M:%S') $*" | tee -a "$LOG"; }

[ -x "$TIDELET/bin/tideletd" ] || { echo "no tideletd at $TIDELET/bin (make build in $TIDELET, or set TIDELET)"; exit 1; }
[ -f "$DRIVER" ] || { echo "no $DRIVER"; exit 1; }
first_task=$(ls -d "$DATA"/*/ | head -1)
WARMUP_IMAGE=${WARMUP_IMAGE:-docker.io/$("$TIDEBOX/.venv/bin/python" -c "import json,sys; print(json.load(open(sys.argv[1]))['container']['image'])" "$first_task/meta.json")}
if [ -n "$TASKS" ]; then select_tasks=($TASKS); else select_tasks=(--all); fi
[ "${SINGLE_CONTAINER:-0}" = 1 ] && select_tasks+=(--single-container)
if [ -n "$TASKS" ]; then ntasks=$(echo $TASKS | wc -w)
elif [ "${SINGLE_CONTAINER:-0}" = 1 ]; then ntasks=$("$TIDEBOX/.venv/bin/python" -c "import json,sys,pathlib; d=pathlib.Path(sys.argv[1]); print(sum(1 for m in d.glob('*/meta.json') if (m.parent / 'tape.json').exists() and not (json.loads(m.read_text()).get('task_spec') or {}).get('sidecars')))" "$DATA")
else ntasks=$(ls "$DATA"/*/meta.json | wc -l); fi

# Only the daemon started from this checkout's bin/ (argv[0] is that path): another tideletd on the
# machine (a development daemon on its own socket and namespace) is left alone.
our_tideletd() { ps -eo pid=,args= > "${TMPDIR:-/tmp}/tideletd.ps.$$"; awk -v b="$TIDELET/bin/tideletd" '$2==b{print $1}' "${TMPDIR:-/tmp}/tideletd.ps.$$"; rm -f "${TMPDIR:-/tmp}/tideletd.ps.$$"; }
stop_tideletd() {
  local pids
  pids=$(our_tideletd)
  if [ -n "$pids" ]; then
    sudo kill $pids
    for _ in $(seq 60); do [ -z "$(our_tideletd)" ] && break; sleep 1; done
    pids=$(our_tideletd); [ -n "$pids" ] && sudo kill -9 $pids
  fi
}

# The runtime's daemons and the shims they start stay off the pool's cores (not persistent).
for unit in containerd.service docker.service; do
  sudo systemctl set-property --runtime "$unit" AllowedCPUs="$DRIVER_CPUS" > /dev/null 2>&1 || log "WARNING could not pin $unit to cpus $DRIVER_CPUS"
done

for spec in $POLICIES; do
  pol=${spec%%:*}
  repeat=1; [ "$spec" != "$pol" ] && repeat=${spec#*:}
  out=$RUN/${OUT_NAME:-$pol${OUT_SUFFIX:-}}   # OUT_NAME: the run's whole directory name, when it does not start with the policy's
  if [ -f "$out/.complete" ]; then log "$pol: already complete, skipping"; continue; fi
  mkdir -p "$out"
  stop_tideletd
  if [ -z "$(swapon --noheadings --show 2>/dev/null)" ]; then
    if sudo swapon "${SWAPFILE:-/swapfile}" 2>/dev/null; then log "$pol: enabled swap on ${SWAPFILE:-/swapfile}"; else log "$pol: WARNING no swap active and ${SWAPFILE:-/swapfile} could not be enabled"; fi
  fi
  log "$pol: swap: $(free -g | awk '/Swap/{print $2 " GB, " $3 " GB used"}')"
  sudo sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches'
  opts=()
  for o in ${POLICY_OPTS:-}; do opts+=(--policy-opt "$o"); done
  daemon=(--policy "$pol" "${opts[@]}" --containerd-namespace "$NAMESPACE" --pool-cgroup auto --pool-cpuset "$POOL_CPUSET" --pool-memory "$POOL_MEMORY"
          --sample-interval "$SAMPLE" --llm-proxy-listen 0.0.0.0:18800 --llm-max-defer "$MAX_DEFER" --log-level debug ${TIDELETD_ARGS:-})
  log "$pol: starting tideletd ${daemon[*]} (on cpus $DRIVER_CPUS)"
  sudo taskset -c "$DRIVER_CPUS" "$TIDELET/bin/tideletd" "${daemon[@]}" > "$out/tideletd.log" 2>&1 &
  for _ in $(seq 60); do curl -fs http://127.0.0.1:7474/healthz > /dev/null 2>&1 && break; sleep 1; done
  if ! curl -fs http://127.0.0.1:7474/healthz > /dev/null 2>&1; then log "$pol: tideletd did not come up; see $out/tideletd.log"; exit 1; fi
  # Leftovers of an earlier attempt (the daemon reloads its store) would block names and ports.
  for sb in $(sudo "$TIDELET/bin/tidelet" list 2>/dev/null | awk 'NR>1{print $1}'); do sudo "$TIDELET/bin/tidelet" rm -f "$sb" > /dev/null 2>&1; done
  # The bridge (and 10.88.0.1, where the controllers and the proxy are reached) exists once a
  # sandbox has been attached; a throwaway sandbox makes it.
  if ! ip -4 addr show tidelet0 2>/dev/null | grep -q 10.88.0.1; then
    if ! warm=$(sudo "$TIDELET/bin/tidelet" run -d --name bridge-warmup --image "$WARMUP_IMAGE" --cpus 1 --memory 256Mi -- sleep 300 2>&1); then
      log "$pol: warm-up sandbox failed to start: $warm"
    fi
    for _ in $(seq 120); do ip -4 addr show tidelet0 2>/dev/null | grep -q 10.88.0.1 && break; sleep 1; done
    sudo "$TIDELET/bin/tidelet" rm -f bridge-warmup > /dev/null 2>&1 || true
  fi
  if ! ip -4 addr show tidelet0 2>/dev/null | grep -q 10.88.0.1; then log "$pol: bridge address 10.88.0.1 not present after the warm-up; stopping"; stop_tideletd; exit 1; fi
  node=$(curl -s http://127.0.0.1:7474/v1alpha1/node)
  "$TIDEBOX/.venv/bin/python" - "$out/run.json" <<EOF
import json, subprocess, sys
def rev(path):
    return subprocess.run(["git", "-C", path, "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
json.dump({"bench": "$BENCH", "data": "$DATA", "policy": "$pol", "repeat": $repeat, "interval": $INTERVAL, "tasks": $ntasks, "sandboxes": $ntasks * $repeat,
           "seed": $SEED, "policy_opts": "${POLICY_OPTS:-}", "tideletd_args": """${daemon[*]}""", "pool_cpuset": "$POOL_CPUSET",
           "pool_memory": "$POOL_MEMORY", "driver_cpus": "$DRIVER_CPUS", "proxy": "$PROXY", "claude_bin": "${CLAUDE_BIN:-}",
           "tidelet_rev": rev("$TIDELET"), "tidebox_rev": rev("$TIDEBOX"), "node": json.loads("""$node""" or "{}"),
           "started_at": "$(date -u +%Y-%m-%dT%H:%M:%SZ)"}, open(sys.argv[1], "w"), indent=1)
EOF
  log "$pol: replaying $ntasks tasks x $repeat through $PROXY into $out (one sandbox offered every $INTERVAL s, seed $SEED), driver on cpus $DRIVER_CPUS${CUTOFF:+, cutoff after $CUTOFF s}"
  started=$(date +%s)
  # CUTOFF: the driver ends the run that many seconds after its first offer (no more sandboxes
  # offered, the running ones stopped; their meta.json says "cutoff": true, no error).
  # stdin from /dev/null: started from a terminal (tmux), sudo's use_pty would hand the driver a pty, and a
  # `tidelet exec -i` that inherits it never returns under a checkpoint-capable daemon.
  sudo -E env PATH="$PATH" taskset -c "$DRIVER_CPUS" "$TIDEBOX/.venv/bin/python" "$DRIVER" --data "$DATA" "${select_tasks[@]}" --policy "$pol" --proxy "$PROXY" </dev/null \
    --repeat "$repeat" --interval "$INTERVAL" --order random --seed "$SEED" --out "$out" --port-base "$PORT_BASE" --admission-timeout "$ADMISSION_TIMEOUT" \
    ${CUTOFF:+--cutoff "$CUTOFF"} ${DRIVER_ARGS:-} > "$out/stdout.log" 2>&1
  rc=$?
  sudo chown -R "$(id -un):$(id -gn)" "$out"
  metas=$(ls "$out"/*/meta.json 2>/dev/null | wc -l)
  complete=$(grep -l '"complete": true' "$out"/*/meta.json 2>/dev/null | wc -l)
  timeouts=$(grep -l '"timed_out": true' "$out"/*/meta.json 2>/dev/null | wc -l)
  errors=$(grep -lE '^ "error": "' "$out"/*/meta.json 2>/dev/null | wc -l)
  curl -s http://127.0.0.1:7474/metrics 2>/dev/null | grep '^tidelet_' > "$out/metrics.txt"
  log "$pol: driver exit $rc after $(( ($(date +%s) - started) / 60 )) min: $metas results, $complete complete compares, $timeouts timed out, $errors errors"
  stop_tideletd
  expected=$((ntasks * repeat))
  if [ -n "${CUTOFF:-}" ]; then
    expected=$metas   # a cut-off run is complete when everything it offered has its record
    log "$pol: cut-off run: $metas sandboxes offered and recorded of $((ntasks * repeat))"
  fi
  if [ "$metas" -ge "$expected" ] && [ "$errors" -eq 0 ] && [ "$metas" -gt 0 ]; then
    touch "$out/.complete"
  else
    log "$pol: NOT complete ($metas of $expected results, $errors errors): stopping here so the cause can be fixed; rerun to continue"
    exit 1
  fi
done
log "all policies done"
