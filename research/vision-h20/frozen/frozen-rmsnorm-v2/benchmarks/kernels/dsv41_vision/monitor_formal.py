# SPDX-License-Identifier: Apache-2.0
"""Record GPU conditions during a separately launched module benchmark."""

import argparse
import datetime
import json
import os
import subprocess
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--watch-pid", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--interval", type=float, default=30)
    parser.add_argument("--max-minutes", type=float, default=45)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    deadline = time.monotonic() + args.max_minutes * 60
    with args.output.open("a") as output:
        while time.monotonic() < deadline:
            try:
                os.kill(args.watch_pid, 0)
            except ProcessLookupError:
                break
            result = subprocess.run(
                [
                    "nvidia-smi",
                    "--id=0,1,2,3",
                    "--query-gpu=index,utilization.gpu,memory.used,temperature.gpu,"
                    "power.draw,clocks.current.sm,clocks.current.memory",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                check=False,
                timeout=15,
            )
            output.write(
                json.dumps(
                    {
                        "utc": datetime.datetime.now(
                            datetime.timezone.utc
                        ).isoformat(),
                        "fields": [
                            "gpu", "util_percent", "memory_mib", "temperature_c",
                            "power_w", "sm_mhz", "memory_mhz",
                        ],
                        "rows": result.stdout.strip().splitlines(),
                        "returncode": result.returncode,
                        "stderr": result.stderr,
                    }
                ) + "\n"
            )
            output.flush()
            time.sleep(args.interval)


if __name__ == "__main__":
    main()
