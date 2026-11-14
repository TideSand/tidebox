#!/bin/bash
# run.sh BENCH/RECORDING PRESET [CUTOFF_SECONDS]: one experiment of the paper. Replays every recorded task
# of data/BENCH/RECORDING at the benchmark's load under the preset's tideletd policy (scripts/presets.sh)
# into output/replay/BENCH/RECORDING/<run name>/, with tideletd confined to a pool cgroup of 56 cores and
# 112 GiB and the host side (tideletd, the driver, the replay controllers, containerd) on the last 8 cores.
#
#   scripts/run.sh algotune/run4-max-50 strict
#   scripts/run.sh algotune/run4-max-50 tidesand
#   scripts/run.sh terminal-bench-4.0/run2-medium overcommit-0.8
#   scripts/run.sh algotune/run4-max-50 tidesand 1800         # a 30-min sample instead of the whole run
#   scripts/run.sh --list                                     # every run the figures need
#
# A run whose directory has a .complete marker is skipped. A run that stopped short (its directory has
# no .complete) is resumed: the tasks already replayed are kept and the rest offered again; to start it
# over, delete the directory first. Environment: POOL_CPUSET (0-55), POOL_MEMORY (112Gi), DRIVER_CPUS
# (56-63) for another machine shape (keep 1 core : 2 GiB), TIDELET (the tidelet checkout, default the
# parent of this TideBox), TASKS (a few task names, for a pilot), NAME (the run's directory instead of the preset's,
# for a pilot: NAME=pilot TASKS="..." scripts/run.sh algotune/run4-max-50 tidesand 600).
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
TIDEBOX=$(cd "$HERE/.." && pwd)
. "$HERE/presets.sh"

if [ "${1:-}" = --list ]; then
  for w in $WORKLOADS; do for p in $MAIN_PRESETS; do echo "$w $p"; done; done
  for p in $ABLATION_PRESETS $SENSITIVITY_PRESETS $REWARD_PRESETS; do echo "algotune/run4-max-50 $p"; done
  exit 0
fi
[ $# -ge 2 ] || { sed -n '2,18p' "$0" | sed 's/^# \{0,1\}//'; exit 2; }
WORKLOAD=${1%/}; PRESET=$2; CUTOFF=${3:-}
BENCH=${WORKLOAD%%/*}; RECORDING=${WORKLOAD#*/}
DATA=$TIDEBOX/data/$BENCH/$RECORDING
[ -d "$DATA" ] || { echo "no recordings at $DATA (README: download the recordings)"; exit 1; }
bench_preset "$BENCH" || { echo "unknown benchmark $BENCH (one of algotune, terminal-bench-4.0, terminal-bench-science)"; exit 1; }
policy_preset "$PRESET" || { echo "unknown preset $PRESET (scripts/presets.sh)"; exit 1; }
REPEAT=${REPEAT_OVERRIDE:-$REPEAT}
RUN_NAME=${NAME:-$RUN_NAME}
OUT=$TIDEBOX/output/replay/$BENCH/$RECORDING/$RUN_NAME
if [ -f "$OUT/.complete" ]; then echo "$OUT is complete; delete it to run again"; exit 0; fi

echo "$BENCH/$RECORDING $PRESET: tideletd --policy $POLICY${POLICY_OPTS:+ ($POLICY_OPTS)}${TIDELETD_ARGS:+ $TIDELETD_ARGS}," \
     "x$REPEAT, one sandbox every $INTERVAL s${CUTOFF:+, cut off after $CUTOFF s} -> $OUT"
export DATA RUN=$TIDEBOX/output/replay/$BENCH/$RECORDING OUT_NAME=$RUN_NAME INTERVAL POLICY_OPTS TIDELETD_ARGS SINGLE_CONTAINER
export DRIVER_ARGS="--name $RUN_NAME${DRIVER_ARGS:+ $DRIVER_ARGS}"
[ -n "$CUTOFF" ] && export CUTOFF || unset CUTOFF
exec "$TIDEBOX/workload/replay_policies.sh" "$POLICY:$REPEAT"
