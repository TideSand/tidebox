#!/bin/bash
# analyze.sh [WORKLOAD...]: the paper's analyses of the replay runs under output/replay/, before scripts/figures.sh.
#
# Per workload, one compare of every run there is against the fidelity reference (the slowest of the five strict
# speed-0 replays of each tool call, 10 % tolerance, 1 s floor, every call counted), throughput over the first 80 % of
# the run's tool calls: output/plots/<bench>/<recording>/compare/bulk80-whole/ (throughput-on-time.csv: one point per run;
# on-time-<run>.csv: every tool call). On AlgoTune Sonnet also the mechanism ablation (compare/mechanisms/: TideSand and
# its single-mechanism runs as one family) and the parameter sensitivity, throughput over the first 90 %
# (compare/sensitivity/). A run that is missing (not run, or not downloaded) is left out and reported.
#
#   scripts/analyze.sh                                  # every workload
#   scripts/analyze.sh algotune/run4-max-50             # one
#
# Each compare reads every tape of its runs once (minutes to an hour per workload; JOBS worker processes, default 8)
# and caches what it read beside the runs, so a second pass is quick.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
TIDEBOX=$(cd "$HERE/.." && pwd)
. "$HERE/presets.sh"
cd "$TIDEBOX"
PY=$TIDEBOX/.venv/bin/python
JOBS=${JOBS:-8}

complete() { ls "$1"/*/meta.json > /dev/null 2>&1; }   # a run: one directory per sandbox, each with its meta.json

# compare WORKLOAD NAME FRACTION RUN... : one compare of the runs that exist (strict and the reference copies first).
compare() {
  local w=$1 name=$2 frac=$3; shift 3
  local base=output/replay/$w ref=output/replay/$w/strict-ref/strict-speed0 out=output/plots/$w/compare/$name
  local runs=() labels=() families="overcommit strict-speed0 agentcgroup" famargs=() f r k
  [ -d "$ref" ] || { echo "-- $w: no fidelity reference at $ref; skipped"; return 0; }
  if complete "$base/strict"; then runs+=(--run "$base/strict/"); labels+=("strict (speed 1)"); else echo "-- $w: strict missing"; fi
  # The reference copies as runs of their own (their throughput is strict's, at speed 0): a flat view of each, since a
  # directory named rep<k> reads as a nested repetition.
  for k in 1 2 3 4 5; do
    local src=$ref/rep$k dst=output/plots/$w/compare/.flat/strict-speed0-rep$k
    complete "$src" || { echo "-- $w: strict-speed0 rep$k missing"; continue; }
    rm -rf "$dst"; mkdir -p "$dst"
    for u in "$src"/*/; do ln -s "$(realpath "$u")" "$dst/$(basename "$u")"; done
    for f in run.json metrics.txt .complete; do [ -e "$src/$f" ] && ln -s "$(realpath "$src/$f")" "$dst/$f"; done
    runs+=(--run "$dst/"); labels+=("strict-speed0 rep$k")
  done
  for r in "$@"; do
    if complete "$base/$r"; then runs+=(--run "$base/$r/"); labels+=("$r"); else echo "-- $w: $r missing"; fi
    # TideSand's single-mechanism runs: one family with TideSand; the sensitivity runs: one family per option swept.
    case $name:$r in
      mechanisms:tidesand-*) f=tidesand;;
      *:tidesand-t*|*:tidesand-bw*|*:tidesand-rb*) f=${r%%[0-9]*}; f=${f%-off};;
      *) f=;;
    esac
    [ -n "$f" ] && case " $families " in *" $f "*) ;; *) families="$families $f";; esac
  done
  for f in $families; do famargs+=(--family "$f"); done
  mkdir -p "$out"
  echo "-- $w: compare $name over ${#labels[@]} runs -> $out"
  nice -n 10 "$PY" workload/plot.py compare --reference "$ref" --strict-label strict "${runs[@]}" --labels "$(IFS=,; echo "${labels[*]}")" \
    --out "$out" --throughput bulk --bulk-fraction "$frac" --reference-stat max --floor-ms 1000 --min-strict-ms 0 \
    "${famargs[@]}" \
    --at-throughput-of "strict-speed0=strict (speed 1)" --no-recording-timeline --jobs "$JOBS" > "$out/compare.log" 2>&1 \
    || { echo "-- $w: compare $name FAILED (see $out/compare.log)"; return 1; }
  tail -n +2 "$out/throughput-on-time.csv" | awk -F, '{printf "   %-22s %9.1f tool calls/hr  %5.1f %% fidelity\n", $1, $3, $4}'
}

for w in ${*:-$WORKLOADS}; do
  w=${w%/}
  runs="overcommit-0.6 overcommit-0.8 overcommit-1.0 tidesand agentcgroup"
  compare "$w" bulk80-whole 0.80 $runs
  if [ "$w" = algotune/run4-max-50 ]; then
    compare "$w" mechanisms 0.80 $runs $ABLATION_PRESETS
    compare "$w" sensitivity 0.90 overcommit-0.6 overcommit-0.8 overcommit-1.0 tidesand $SENSITIVITY_PRESETS
  fi
done
echo "done; next: scripts/figures.sh"
