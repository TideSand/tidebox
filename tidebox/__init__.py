"""TideBox: replay a recorded agent run (a tape) in a sandbox without a model.

Components (one subpackage each):

    api         the tape format; stdlib only
    controller  host side, replay: plays the model, answering a real harness's model calls from the tape
    telemetry   the sandbox's CPU and memory every 250 ms, as CSV next to the replay tape
"""

__version__ = "0.1.0"
