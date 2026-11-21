#!/usr/bin/env python3
"""Mirror per-step loss over an existing SSH connection and refresh a local PNG."""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
import time
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", required=True)
    parser.add_argument("--port", required=True, type=int)
    parser.add_argument("--host-key-alias", required=True)
    parser.add_argument("--remote-run", required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    log = args.output / "metrics.jsonl"
    plot = args.output / "native_sft_loss_live.png"
    renderer = Path(__file__).with_name("plot_native_sft_loss.py")
    command = [
        "ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", "-o",
        f"HostKeyAlias={args.host_key_alias}", "-o", "HostKeyAlgorithms=ssh-ed25519",
        "-p", str(args.port), args.host,
        f"cd {shlex.quote(args.remote_run)} || exit 1; "
        'if test -f full/training.json; then echo complete; '
        'elif test -f status && test "$(cat status)" != 0; then echo failed; '
        'else echo running; fi; '
        'if test -s full/metrics.jsonl; then cat full/metrics.jsonl; fi',
    ]
    previous = log.read_bytes() if log.is_file() and plot.is_file() else b""
    print("Watching", f"{args.host}:{args.remote_run}/full/metrics.jsonl", flush=True)
    while True:
        try:
            result = subprocess.run(command, capture_output=True, timeout=25)
        except subprocess.TimeoutExpired:
            print("SSH read timed out; retrying", flush=True)
        else:
            if result.returncode:
                print(result.stderr.decode(errors="replace").strip(), flush=True)
            else:
                phase, _, data = result.stdout.partition(b"\n")
                if data and data != previous:
                    rows = [json.loads(line) for line in data.splitlines()]
                    pending = log.with_name(".metrics.jsonl.incoming")
                    pending.write_bytes(data)
                    subprocess.run([sys.executable, str(renderer), str(pending), str(plot)], check=True)
                    pending.replace(log)
                    previous = data
                    print(f"step={rows[-1]['step']} loss={rows[-1]['loss']}", flush=True)
                if phase.strip() in {b"complete", b"failed"}:
                    if phase.strip() == b"failed":
                        raise SystemExit("Formal training stopped early; recorded loss points were preserved")
                    if not plot.is_file():
                        raise SystemExit("Training completed without a verified loss curve")
                    print("Full training curve synced", flush=True)
                    return
        time.sleep(15)


if __name__ == "__main__":
    main()

