"""Play a sound when a long scrape finishes.

Two modes. Waiting on a process id is for a job already running, in this
terminal or another:

    uv run python scripts/beep_when_done.py --pid 1064

Wrapping the command is better when you have not started it yet, because the
exit code is known and the tone says whether it worked:

    uv run python scripts/beep_when_done.py -- just scrape-info -- --site 5,14

Find the pid of a running pipeline with:

    powershell -NoProfile -Command "Get-CimInstance Win32_Process |
      Where-Object { $_.CommandLine -match 'just.exe. scrape' } |
      ForEach-Object { \"$($_.ProcessId) :: $($_.CommandLine)\" }"

Watch the top-level `just` process, not the python one: the pipeline replaces
its python child at every step, so a beep on that would fire after rooms and
again after prices.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time

DONE = ((660, 200), (880, 400))  # exit 0, or the watched process is gone
FAILED = ((440, 300), (330, 600))  # non-zero exit


def beep(tones: tuple[tuple[int, int], ...]) -> None:
    try:
        import winsound

        for freq, ms in tones:
            winsound.Beep(freq, ms)
    except (ImportError, RuntimeError):
        # Not Windows, or no sound device: the terminal bell still carries.
        for _ in tones:
            print("\a", end="", flush=True)
            time.sleep(0.3)


def wait_for_pid(pid: int) -> None:
    """Block until the process exits. No polling: the kernel wakes us."""
    import ctypes

    SYNCHRONIZE, INFINITE, WAIT_FAILED = 0x00100000, 0xFFFFFFFF, 0xFFFFFFFF
    kernel32 = ctypes.windll.kernel32
    handle = kernel32.OpenProcess(SYNCHRONIZE, False, pid)
    if not handle:
        raise SystemExit(f"No process {pid} (already finished?)")
    try:
        if kernel32.WaitForSingleObject(handle, INFINITE) == WAIT_FAILED:
            raise SystemExit(f"Could not wait on {pid}")
    finally:
        kernel32.CloseHandle(handle)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pid", type=int, help="wait for this process to exit")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()

    if args.command:
        command = args.command[1:] if args.command[0] == "--" else args.command
        code = subprocess.run(command).returncode
        print(f"\n{' '.join(command)} exited {code}")
    elif args.pid:
        started = time.monotonic()
        print(f"Waiting for pid {args.pid} ...", flush=True)
        wait_for_pid(args.pid)
        # An exit code we did not spawn is not ours to read, so the tone only
        # says "over", never "fine".
        print(f"pid {args.pid} finished after {time.monotonic() - started:.0f}s")
        code = 0
    else:
        parser.error("pass --pid N, or a command after --")

    beep(DONE if code == 0 else FAILED)
    sys.exit(code)


if __name__ == "__main__":
    main()
