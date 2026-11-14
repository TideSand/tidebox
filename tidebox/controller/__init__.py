"""Replay, host side: plays the model.

The controller reads a tape and answers model calls from it: a real harness (Claude Code) runs in the
sandbox pointed at the controller as if it were the provider; the controller matches each request to a
recorded step and streams the raw recorded response at the recorded pace. The harness performs the tool
calls itself, for real, in the sandbox. With `--output <file>` the controller also records the replay as a
tape (`api.recorder`): the responses it served and the tool results it got back, with the replay's own
timing and `meta.replay_of` set to the recording's id. That file is the input to `compare`.

    simulator.py     the provider-shaped endpoint
    server.py        hosts it, plus GET /tidebox/state
    cursor.py        position in the tape per thread; advances, forks for subagents
    matcher.py       request -> recorded step (by tool-call ids, prompt, opening message, tool count)
    pacing.py        recorded latency, scaled by --speed
    identifiers.py   remaps run-specific ids (session, background task, subagent) between
                     the recording and the replay
    compare.py       recording vs. replay tape, step by step; divergence report
"""
