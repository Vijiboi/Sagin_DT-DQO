from __future__ import annotations

import argparse
import json

from env.config import SimulationConfig
from sim.runner import SimulationRunner


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Classical DT-DQO SAGIN simulator")
    parser.add_argument("--slots", type=int, default=8, help="Number of simulation slots")
    parser.add_argument("--seed", type=int, default=7, help="Random seed")
    parser.add_argument("--reads", type=int, default=5, help="Classical annealing reads per AP")
    parser.add_argument("--sweeps", type=int, default=20, help="Classical annealing sweeps per AP")
    parser.add_argument("--output-dir", type=str, default="results", help="Directory for CSV/JSON outputs")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = SimulationConfig(
        slots=args.slots,
        seed=args.seed,
        anneal_reads=args.reads,
        anneal_sweeps=args.sweeps,
        output_dir=args.output_dir,
    )
    runner = SimulationRunner(config)
    _, summary, output_path = runner.run()
    print(json.dumps(summary, indent=2))
    print(f"Outputs saved in: {output_path}")


if __name__ == "__main__":
    main()
