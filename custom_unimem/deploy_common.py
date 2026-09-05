"""ROS-free helpers shared by the two UniMem deploy nodes.

Kept out of the node files so both robots get identical prompt handling and identical
keyboard controls, and so this can be unit-tested without a ROS installation.
"""

from __future__ import annotations

from collections.abc import Callable
import pathlib
import select
import sys
import termios
import threading

import yaml


def load_tasks(tasks_file: str, warn: Callable[[str], None]) -> list[str]:
    """Ordered ``tasks:`` list from a YAML file; ``[]`` if unset, missing or empty."""
    if not tasks_file:
        return []
    path = pathlib.Path(tasks_file)
    if not path.is_file():
        warn(f"tasks_file '{tasks_file}' not found; falling back to the 'prompt' parameter.")
        return []
    try:
        doc = yaml.safe_load(path.read_text()) or {}
    except yaml.YAMLError as e:
        warn(f"failed to parse tasks_file '{tasks_file}': {e}")
        return []
    tasks = doc.get("tasks") if isinstance(doc, dict) else doc
    if not isinstance(tasks, list) or not tasks:
        warn(f"tasks_file '{tasks_file}' has no non-empty 'tasks:' list.")
        return []
    return [str(t) for t in tasks]


class KeyListener:
    """Single-keypress listener on stdin, running on its own daemon thread.

    Canonical mode and echo are disabled so keys arrive without Enter, but ISIG is left
    on so Ctrl+C still reaches the process. Terminal settings are always restored, even
    if the thread dies on an exception.
    """

    def __init__(self, keymap: dict[str, Callable[[], None]], warn: Callable[[str], None]) -> None:
        self._keymap = {k.lower(): v for k, v in keymap.items()}
        self._warn = warn
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> bool:
        if not sys.stdin.isatty():
            self._warn("stdin is not a TTY; keyboard controls are disabled.")
            return False
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return True

    def stop(self) -> None:
        self._stop.set()

    def _run(self) -> None:
        fd = sys.stdin.fileno()
        old = termios.tcgetattr(fd)
        try:
            new = termios.tcgetattr(fd)
            new[3] = new[3] & ~(termios.ICANON | termios.ECHO)
            termios.tcsetattr(fd, termios.TCSANOW, new)
            while not self._stop.is_set():
                ready, _, _ = select.select([sys.stdin], [], [], 0.2)
                if not ready:
                    continue
                handler = self._keymap.get(sys.stdin.read(1).lower())
                if handler is not None:
                    handler()
        except Exception as e:  # never let a terminal quirk take down the node
            self._warn(f"key listener disabled: {e}")
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old)
