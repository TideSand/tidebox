"""`python -m tidebox.controller`: same as `tidebox replay`."""

from tidebox.cli import main

raise SystemExit(main(["replay", *__import__("sys").argv[1:]]))
