# TideBox

TideBox replays recorded agent runs in sandboxes without calling a model, and analyses the replays. It is the
replay harness of the TideSand paper, and this repository holds what is needed to reproduce the paper's results with
[tidelet](..), the sandbox daemon TideSand runs in.

A recording (a *tape*) holds every model request and response of one agent run, with its timing. To replay it,
the real harness (Claude Code, the version that made the recording) runs inside a fresh sandbox with the task's
environment image, and its model calls are answered from the tape by a per-sandbox *controller*, paced as recorded.
The tool calls run for real in the sandbox. So every policy sees the same agent behaviour, and only the sandbox
system differs.

```
tidebox/                 the tape format (api/) and the replay controller (controller/), `tidebox replay --llm`
workload/replay.py       the replay driver: offers sandboxes at a fixed rate, one controller per sandbox
workload/plot.py         the analysis: `compare` (throughput and tool-call fidelity of runs against the reference)
workload/replaykit/      the driver's and the analysis' code; plot/paper/ draws the paper's figures
scripts/                 setup.sh, presets.sh, run.sh, analyze.sh, figures.sh: the steps below
data/  images/  output/  downloaded (not in git): recordings, task images, replay results
```

## What reproducing takes

- **A host like the evaluation's**: an arm64 bare-metal machine with 64 cores and 128 GB of DRAM (AWS
  `c7g.metal`), Ubuntu 24.04 with Linux 6.16 or later, cgroup v2, sudo. The sandboxes get a pool of 56 cores and
  112 GiB; the last 8 cores run the daemon, the driver and the controllers. Another machine shape works if the pool
  keeps 1 core per 2 GiB (`POOL_CPUSET`, `POOL_MEMORY`, `DRIVER_CPUS`; see `scripts/run.sh`), but the numbers will
  differ from the paper's.
- **Disk**: about 25 GB of recordings, 100 GB of image archives plus the loaded images, a 64 GB swap file, and up to
  350 GB for the replay results of every run (plus room for the download while it unpacks). 1 TB is comfortable.
- **Time**: the 81 runs take about 250 host-hours in total (below, per run). Runs are independent; several hosts
  can share them out, each writing its own `output/replay/<bench>/<recording>/<run>/`.
- **No model access**: replays answer every model call from the tapes; nothing reaches a model provider.

To check the figures without running anything, download the released results (step 2) and skip to step 5.

## 1. Get the code

TideBox lives inside the tidelet checkout:

```sh
git clone https://github.com/TideSand/tidelet.git
cd tidelet
git clone https://github.com/TideSand/tidebox.git
cd tidebox
```

## 2. Download the data

Three Hugging Face datasets, each unpacked into its own folder of this checkout (all three folders are in
`.gitignore`). The datasets hold one zstd tarball per recording, per replay run and per task-directory set;
`scripts/download.sh` downloads one (resumable: run it again) and unpacks it:

```sh
pip install -U huggingface_hub            # the `hf` command; tar and zstd are needed too
scripts/download.sh recordings            # TideSand/tidebox-recordings -> data/            (~5 GB download)
scripts/download.sh images                # TideSand/tidebox-images     -> images/          (~100 GB)
scripts/download.sh results               # TideSand/tidebox-results    -> output/replay/   (~65 GB, 350 GB unpacked; optional)
```

| folder | content |
|---|---|
| `data/<bench>/<recording>/<task>/` | one recorded agent run: `tape.json` (the tape), `tape.telemetry.csv` (the sandbox's CPU and memory every 250 ms), `meta.json` (the task's limits and image, the Claude Code version, the verifier's reward), `harbor/` (the benchmark harness's trial directory) |
| `images/<set>/` | per image set (`algotune`, `tb4`, `tbsci`): a `docker save` archive in parts, its checksums, the list of images, and the task directories (`tasks-arm64/<task>/`, whose `tests/` the reward experiment runs) |
| `output/replay/<bench>/<recording>/<run>/` | the paper's replay runs (step 4's output), to analyse without running them |

The recordings, with Claude Sonnet 5.5 and DeepSeek V4.1 Flash on Claude Code 2.1.300-2.1.309:

| workload | Sonnet | DeepSeek |
|---|---|---|
| AlgoTune (50 tasks, 8 CPUs / 16 GB each) | `algotune/run4-max-50` | `algotune/run1-deepseek-v4.1-flash` |
| Terminal-Bench 4.0 (50 single-container tasks, 2-16 CPUs / 4-16 GB) | `terminal-bench-4.0/run2-medium` | `terminal-bench-4.0/run1-deepseek-v4.1-flash` |
| Terminal-Bench Science (50 tasks, 1-4 CPUs / 2-16 GB) | `terminal-bench-science/run1-rand50` | `terminal-bench-science/run1-deepseek-v4.1-flash` |

and two more AlgoTune recordings for the motivation (Fig. 5b), the same ten tasks recorded under strict reservation
(`algotune/run2-max`) and under overcommit (`algotune/run3-max-loadgate`).

## 3. Set up the host

```sh
scripts/setup.sh prepare     # packages, Docker, Go, tidelet (make build), CNI, CRIU, swap, the venv,
                             # the recordings' Claude Code versions, and `docker load` of images/
scripts/setup.sh check       # every item ok, or MISSING with what to do
```

`prepare` builds tidelet in the parent directory (`make build`: `bin/tideletd`, `bin/tidelet`, `bin/tidelet-init`),
installs CRIU 4.1 from source with `scripts/criu-sockets.patch` (needed on Linux 6.16 and later), switches Docker to
containerd's image store (tideletd runs the task images from containerd's `moby` namespace), and installs each
Claude Code version the recordings were made with under `~/.local/share/claude/versions/` (the driver bind-mounts the
recording's version into its sandbox). Loading the images needs about as much free disk again as `images/`.

## 4. Run the experiments

One command per run; each replays every recorded task of a workload under one policy preset:

```sh
scripts/run.sh --list                                   # the 81 runs the figures need
scripts/run.sh algotune/run4-max-50 strict-speed0-rep1  # one run
scripts/run.sh --list | while read w p; do scripts/run.sh "$w" "$p" || break; done   # all of them, in order
```

`run.sh` starts `tideletd` with the preset's policy and the LLM proxy, confined to the 56-core / 112 GiB pool cgroup,
offers the recorded tasks at the workload's rate (`workload/replay_policies.sh`, `workload/replay.py`), and stops the
daemon at the end. The output goes to `output/replay/<bench>/<recording>/<run>/`: one directory per sandbox (the
replay tape, the sandbox's telemetry, `meta.json`, Claude Code's and the controller's logs) and, beside them,
`tideletd.log`, `run.json` (the setup), `metrics.txt` and the driver's logs. A finished run has a `.complete` marker
and is skipped next time; a run that stopped short is resumed where it stopped (delete its directory to start over).

The presets (`scripts/presets.sh`):

| run | policy | used in |
|---|---|---|
| `strict-speed0-rep1` ... `rep5` | strict reservation, one copy per task, model answering at once (`--speed 0`); the five copies are the **fidelity reference** (`strict-ref/strict-speed0/rep<k>/`) | every figure |
| `strict` | strict reservation, one copy per task, the recorded pace | Figs. 3, 7, 9, 12 |
| `overcommit-0.6`, `-0.8`, `-1.0` | threshold overcommit (admit while measured CPU and memory are under theta) | Figs. 7, 9, 10, 12 |
| `tidesand` | TideSand, default parameters (alpha 0.7, beta 0.8, rollback threshold 0.85), whole-container checkpoints | Figs. 7, 9, 10, 11, 12, 13 |
| `agentcgroup` | the AgentCgroup baseline, no admission control | Figs. 7, 10 |
| AlgoTune Sonnet only: `tidesand-defer`, `-speculate`, `-rollback`, `agentcgroup-g0.8` | one TideSand mechanism at a time; AgentCgroup behind overcommit's gate at 0.8 | Fig. 10 |
| AlgoTune Sonnet only: `tidesand-t0.55`, `-t0.85`, `-bw-off`, `-bw0.6`, `-bw1.0`, `-rb-off`, `-rb0.7`, `-rb1.0` | one TideSand parameter moved: CPU budget alpha, bandwidth budget beta, rollback threshold | Fig. 11 |
| AlgoTune Sonnet only: `strict-verify`, `overcommit-0.8-verify`, `tidesand-verify` | the policy's run with each task's verifier run in the sandbox afterwards, two copies per task | Fig. 8 |

Loads: AlgoTune x2 (100 sandboxes, one every 15 s), Terminal-Bench 4.0 x8 (400 sandboxes, one every 3 s),
Terminal-Bench Science x2 (100 sandboxes, one every 15 s); strict runs one copy per task. On Terminal-Bench Science a
sandbox may swap up to its memory limit, as in the recording (strict `swap=true`, the others
`tideletd --sandbox-swap=limit`). Every task is capped at two hours of agent time.

Durations on the evaluation host, in hours (a speed-0 copy; five per workload):

| workload | strict | speed-0 copy | overcommit 0.6 / 0.8 / 1.0 | tidesand | agentcgroup |
|---|---|---|---|---|---|
| AlgoTune Sonnet | 7.2 | 1.7 | 3.1 / 2.6 / 2.5 | 2.0 | 2.4 |
| Terminal-Bench 4.0 Sonnet | 2.1 | 1.7 | 2.6 / 0.6 / 2.3 | 2.4 | 2.3 |
| Terminal-Bench Science Sonnet | 6.0 | 4.0 | 4.9 / 5.0 / 4.9 | 5.4 | 4.9 |
| AlgoTune DeepSeek | 4.6 | 2.9 | 3.2 / 3.1 / 2.8 | 3.1 | 2.9 |
| Terminal-Bench 4.0 DeepSeek | 3.5 | 1.6 | 4.8 / 4.6 / 4.4 | 6.3 | 6.0 |
| Terminal-Bench Science DeepSeek | 5.4 | 4.1 | 5.5 / 6.6 / 5.5 | 5.5 | 5.5 |

The AlgoTune Sonnet extras take about 2 hours each, `strict-verify` 14 hours.

A short pilot first is a good check of the host: `NAME=pilot TASKS="algotune-btsp algotune-kd-tree" scripts/run.sh
algotune/run4-max-50 tidesand 600` replays two tasks under TideSand for ten minutes into
`output/replay/algotune/run4-max-50/pilot/` (delete it afterwards).

## 5. Analyse

```sh
scripts/analyze.sh                          # every workload; or name one: scripts/analyze.sh algotune/run4-max-50
```

Per workload, one `workload/plot.py compare` of every run there is against the fidelity reference, into
`output/plots/<bench>/<recording>/compare/bulk80-whole/` (`throughput-on-time.csv`: one point per run;
`on-time-<run>.csv`: every tool call; `summary.csv`; figures over time). On AlgoTune Sonnet also the mechanism
ablation, `compare/mechanisms/`, and the parameter sensitivity, `compare/sensitivity/`. The definitions, as in the paper:

- **Tool-call fidelity**: the share of tool calls whose execution time is at most 10 % above their reference time,
  the slowest of the five strict speed-0 copies of that call (floored at 1 s; every call counted). Execution is the
  gap between the model response the agent received and its next request; a response the LLM proxy held back is not
  execution, and the hold is subtracted (from the proxy's log in `tideletd.log` and the controllers' logs).
- **Throughput**: tool calls completed per hour, until 80 % of the run's tool calls have completed (90 % for the
  sensitivity sweep).

A compare reads every tape of its runs once (minutes to an hour per workload) and caches what it read under
`~/.cache/tidebox/`, so a second pass is quick.

## 6. Draw the figures

```sh
scripts/figures.sh       # -> output/plots/paper/
```

| paper | file | from |
|---|---|---|
| Fig. 3 | `cpu-memory.pdf` | two AlgoTune Sonnet tasks' sandbox CPU and memory under `strict` |
| Fig. 5a | `algotune-tool-resource.pdf` | ten AlgoTune tool calls replayed ten times each under smaller allocations (`output/replay/algotune/run2-max/variability/measurements.csv`, released as data) |
| Fig. 5b | `algotune-overcommit-run.pdf` | the rewards of the recordings `algotune/run2-max` (strict) and `run3-max-loadgate` (overcommit) |
| Fig. 7 | `throughput-fidelity-2x3.pdf` | `compare/bulk80-whole` of the six workloads |
| Fig. 8 | `policies-algotune-replay-reward.pdf` | the `*-verify` runs' rewards |
| Fig. 9 | `policies-tbs-resources.pdf` | the telemetry of `strict`, `overcommit-0.8`, `tidesand` on Terminal-Bench Science Sonnet |
| Fig. 10 | `tidesand-mechanisms.pdf` | `compare/mechanisms` of AlgoTune Sonnet |
| Fig. 11 | `tidesand-ablation.pdf` | `compare/sensitivity` of AlgoTune Sonnet |
| Fig. 12 | `policies-tool-call-distribution.pdf` | the per-call data of `compare/bulk80-whole`, Sonnet workloads |
| Fig. 13 | `rollback-micro.pdf` | the `tidesand` runs' `tideletd.log` (checkpoints, rollbacks) and `metrics.txt` |
| | `workload-burstiness.pdf` | the telemetry of `strict` on the six workloads: each sandbox's peak-to-mean CPU and the length of its bursts (runs of consecutive seconds at 0.5 cores or more that reach 4 cores) |

Each script also runs on its own from this directory, e.g. `.venv/bin/python
workload/replaykit/plot/paper/throughput_2x3.py`; its docstring says what it reads.

## Using the pieces directly

```sh
# replay a few tasks under whatever policy the running tideletd has, through its LLM proxy
.venv/bin/python workload/replay.py --data data/algotune/run4-max-50 algotune-btsp --policy tidesand \
    --proxy http://10.88.0.1:18800 --repeat 2 --interval 15
# one controller by hand: serve a tape as the model on :8081, record the replay
.venv/bin/tidebox replay --llm --tape data/algotune/run4-max-50/algotune-btsp/tape.json --listen :8081 --output replay.json
.venv/bin/tidebox compare data/algotune/run4-max-50/algotune-btsp/tape.json replay.json
```

`workload/replay.py --help` and `workload/plot.py compare --help` list every option.

## Tests

```sh
uv pip install --python .venv/bin/python -e '.[workload,dev]'
.venv/bin/pytest
```

## License

Apache 2.0.
