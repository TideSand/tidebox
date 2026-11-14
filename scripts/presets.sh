# presets.sh: what every experiment of the paper means for tideletd and for the replay driver.
# Sourced by scripts/run.sh; nothing here runs anything.
#
# A run is (workload, preset). The workload is a benchmark and one of its recordings; the preset is a
# tideletd policy with its options and daemon flags, and the run's name, which is the directory the
# replay goes to: output/replay/<bench>/<recording>/<run name>/.

# The six workloads of the evaluation: <bench>/<recording> (rows: Claude Sonnet 5.5, DeepSeek V4.1 Flash).
WORKLOADS="algotune/run4-max-50 terminal-bench-4.0/run2-medium terminal-bench-science/run1-rand50
algotune/run1-deepseek-v4.1-flash terminal-bench-4.0/run1-deepseek-v4.1-flash terminal-bench-science/run1-deepseek-v4.1-flash"

# The presets every workload runs (the 2x3 figure and the figures drawn from the same runs).
MAIN_PRESETS="strict strict-speed0-rep1 strict-speed0-rep2 strict-speed0-rep3 strict-speed0-rep4 strict-speed0-rep5
overcommit-0.6 overcommit-0.8 overcommit-1.0 tidesand agentcgroup"
# AlgoTune with Claude Sonnet 5.5 only (algotune/run4-max-50): the mechanism ablation, the parameter
# sensitivity, and the replay-reward experiment.
ABLATION_PRESETS="tidesand-defer tidesand-speculate tidesand-rollback agentcgroup-g0.8"
SENSITIVITY_PRESETS="tidesand-t0.55 tidesand-t0.85 tidesand-bw-off tidesand-bw0.6 tidesand-bw1.0 tidesand-rb-off tidesand-rb0.7 tidesand-rb1.0"
REWARD_PRESETS="strict-verify overcommit-0.8-verify tidesand-verify"

# bench_preset BENCH: INTERVAL (seconds between sandboxes offered), REPEAT (copies of every recorded
# task offered: the load) and SINGLE_CONTAINER (1: only tasks without service containers).
bench_preset() {
  case $1 in
    algotune)               INTERVAL=15; REPEAT=2; SINGLE_CONTAINER=0;;   # 50 tasks x2: 100 sandboxes, 8 CPUs / 16 GB each
    terminal-bench-4.0)     INTERVAL=3;  REPEAT=8; SINGLE_CONTAINER=1;;   # 50 single-container tasks x8: 400 sandboxes, 2-16 CPUs / 4-16 GB
    terminal-bench-science) INTERVAL=15; REPEAT=2; SINGLE_CONTAINER=0;;   # 50 tasks x2: 100 sandboxes, 1-4 CPUs / 2-16 GB
    *) return 1;;
  esac
}

# policy_preset NAME: POLICY (tideletd --policy), POLICY_OPTS (--policy-opt pairs), TIDELETD_ARGS (other
# daemon flags), DRIVER_ARGS (replay.py flags), RUN_NAME (the output directory, relative to the
# recording's) and REPEAT_OVERRIDE (a preset that runs at its own load). Uses BENCH for the
# Terminal-Bench Science swap rule below.
policy_preset() {
  POLICY_OPTS=""; TIDELETD_ARGS=""; DRIVER_ARGS=""; REPEAT_OVERRIDE=""; RUN_NAME=$1
  case $1 in
    strict)
      # No overcommit: a sandbox runs only when its limits fit. One copy per task, at the recorded pace.
      POLICY=strict; REPEAT_OVERRIDE=1;;
    strict-speed0-rep[1-5])
      # The fidelity reference: five strict replays with the model answering at once (--speed 0); a
      # tool call's reference time is the slowest of its five executions.
      POLICY=strict; REPEAT_OVERRIDE=1; DRIVER_ARGS="--speed 0"; RUN_NAME=strict-ref/strict-speed0/rep${1#strict-speed0-rep};;
    overcommit-0.6|overcommit-0.8|overcommit-1.0)
      # Threshold overcommit: a sandbox starts while the pool's measured CPU and memory are under theta.
      POLICY=overcommit; POLICY_OPTS="threshold=${1#overcommit-}";;
    tidesand)
      # TideSand with its defaults (CPU budget alpha 0.7, bandwidth budget beta 0.8, rollback threshold
      # 0.85); whole-container checkpoints for selective rollback.
      POLICY=tidesand; TIDELETD_ARGS="--container-checkpoints";;
    agentcgroup)
      # The AgentCgroup baseline: tool-call resource domains, no admission control.
      POLICY=agentcgroup;;
    agentcgroup-g0.8)
      # AgentCgroup behind overcommit's admission rule at 0.8.
      POLICY=agentcgroup; POLICY_OPTS="gate=0.8";;
    tidesand-defer|tidesand-speculate|tidesand-rollback)
      # One mechanism at a time: agent-aware scheduling alone (defer), speculative swapping alone,
      # selective rollback alone.
      POLICY=tidesand; POLICY_OPTS="mechanisms=${1#tidesand-}"
      [ "$1" = tidesand-rollback ] && TIDELETD_ARGS="--container-checkpoints";;
    tidesand-t0.55|tidesand-t0.85)
      # The CPU budget alpha.
      POLICY=tidesand; POLICY_OPTS="target=${1#tidesand-t}"; TIDELETD_ARGS="--container-checkpoints";;
    tidesand-bw-off|tidesand-bw0.6|tidesand-bw1.0)
      # The memory-bandwidth budget beta (off: traffic not gated).
      POLICY=tidesand; TIDELETD_ARGS="--container-checkpoints"
      case $1 in tidesand-bw-off) POLICY_OPTS="bandwidth=0";; *) POLICY_OPTS="bwTarget=${1#tidesand-bw}";; esac;;
    tidesand-rb-off|tidesand-rb0.7|tidesand-rb1.0)
      # The rollback threshold: the share of the machine's cores or bandwidth that counts as saturated
      # (off: no call is checkpointed, so none is rolled back).
      POLICY=tidesand; TIDELETD_ARGS="--container-checkpoints"
      case $1 in tidesand-rb-off) POLICY_OPTS="slots=0";; *) POLICY_OPTS="pressure=${1#tidesand-rb}";; esac;;
    strict-verify|overcommit-0.8-verify|tidesand-verify)
      # The replay-reward experiment: the policy's run with the task's own verifier after the agent,
      # two copies of every task (strict included: its second copy is the verifier's noise floor).
      policy_preset "${1%-verify}" || return 1   # the policy's own preset, Terminal-Bench Science rule included
      RUN_NAME=$1; REPEAT_OVERRIDE=2; DRIVER_ARGS="--verify"; return 0;;
    *) return 1;;
  esac
  # Terminal-Bench Science was recorded under `docker run --memory` without --memory-swap: a task may
  # swap up to its memory limit. Strict gives a sandbox that swap; the other policies get the same cap
  # from the daemon (--sandbox-swap=limit), the policy untouched.
  if [ "${BENCH:-}" = terminal-bench-science ]; then
    if [ "$POLICY" = strict ]; then POLICY_OPTS="${POLICY_OPTS:+$POLICY_OPTS }swap=true"
    else TIDELETD_ARGS="${TIDELETD_ARGS:+$TIDELETD_ARGS }--sandbox-swap=limit"; fi
  fi
  return 0
}
