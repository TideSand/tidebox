#!/bin/bash
# figures.sh: every figure of the paper's evaluation and motivation into output/plots/paper/, from the replay runs under
# output/replay/, the compares scripts/analyze.sh wrote under output/plots/, and the recordings under data/.
# A figure whose inputs are missing fails alone; the others are still drawn.
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
cd "$HERE/.."
PY=${PY:-.venv/bin/python}; P=workload/replaykit/plot/paper
mkdir -p output/plots/paper
failed=0
fig() { echo "== $*"; "$PY" "$@" || { echo "   FAILED: $*"; failed=$((failed + 1)); }; }

# Evaluation
fig $P/throughput_2x3.py bulk80-whole                    # throughput vs tool-call fidelity, six workloads
fig $P/policies_algotune_replay_reward.py                # task reward error under each policy (AlgoTune Sonnet)
fig $P/policies_resources.py                             # host CPU, DRAM and swap over time (Terminal-Bench Science Sonnet)
fig $P/tidesand_mechanisms.py                            # one mechanism at a time (AlgoTune Sonnet)
fig $P/tidesand_ablation.py                              # parameter sensitivity (AlgoTune Sonnet)
fig $P/policies_tool_call_distribution.py bulk80-whole   # tool-call slowdown distributions (Sonnet workloads)
fig $P/rollback_micro.py                                 # how often selective rollback acts and what it costs
fig $P/workload_burstiness.py                            # spikiness and burst length of each workload's own CPU demand (strict)
# Motivation
fig $P/cpu_memory.py output/replay/algotune/run4-max-50/strict output/plots/paper/cpu-memory.pdf algotune-sylvester-solver algotune-minimum-volume-ellipsoid
fig $P/algotune_overcommit_run.py 10                     # AlgoTune rewards: a strict vs an overcommitted recording
fig $P/algotune_tool_resource.py                         # tool-call slowdown under smaller allocations
echo "figures in output/plots/paper/ ($failed failed)"
[ "$failed" -eq 0 ]
