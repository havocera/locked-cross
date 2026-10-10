"""Resume the prespecified ten-seed internal benchmark sequentially."""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys


SEEDS = (104729, 130363, 155921, 181081, 205759, 231481, 257053, 282433, 308113, 333667)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--responses", type=Path, required=True)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    worker = Path(__file__).with_name("run_internal_seed.py")
    for seed in SEEDS:
        expected = args.output / f"metrics_seed_{seed}.csv"
        if expected.is_file() and expected.stat().st_size > 0:
            print(f"SKIP seed {seed}: output exists", flush=True)
            continue
        command = [
            sys.executable, str(worker),
            "--responses", str(args.responses),
            "--features", str(args.features),
            "--output", str(args.output),
            "--seed", str(seed),
        ]
        print(f"START seed {seed}", flush=True)
        completed = subprocess.run(command, check=False)
        print(f"END seed {seed} exit={completed.returncode}", flush=True)
        if completed.returncode != 0:
            raise SystemExit(completed.returncode)


if __name__ == "__main__":
    main()
