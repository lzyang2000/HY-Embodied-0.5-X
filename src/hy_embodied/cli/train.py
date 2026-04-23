"""Command-line entry point for SFT training.

Usage::

    python -m hy_embodied.cli.train --config configs/sft/example_small.yaml
"""

from __future__ import annotations

import argparse
import logging

from hy_embodied.training import SFTConfig, run_sft


def _setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s %(name)s: %(message)s",
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SFT trainer for HY-Embodied-0.5-X.")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to a JSON/YAML SFT config file (see configs/sft/).",
    )
    return parser.parse_args()


def main() -> None:
    _setup_logging()
    args = _parse_args()
    config = SFTConfig.from_file(args.config)
    run_sft(config)


if __name__ == "__main__":
    main()
