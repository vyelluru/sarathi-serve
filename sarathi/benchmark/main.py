import argparse
import logging
import os

import yaml

from sarathi.benchmark.benchmark_runner import BenchmarkRunnerLauncher
from sarathi.benchmark.config import BenchmarkConfig
from sarathi.benchmark.constants import LOGGER_FORMAT, LOGGER_TIME_FORMAT
from sarathi.benchmark.utils.random import set_seeds
from sarathi.logger import init_logger

logger = init_logger(__name__)


def _load_benchmark_config() -> BenchmarkConfig:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to a YAML benchmark config file.",
    )
    parser.add_argument("-h", "--help", action="store_true")
    config_args, remaining_args = parser.parse_known_args()

    if config_args.help:
        parser.print_help()
        if config_args.config is None:
            print()
            BenchmarkConfig.create_from_cli_args(args=["--help"])
        raise SystemExit(0)

    if config_args.config is None:
        return BenchmarkConfig.create_from_cli_args()

    if remaining_args:
        raise ValueError("--config cannot be combined with additional CLI overrides.")

    with open(config_args.config, "r", encoding="utf-8") as f:
        raw_config = yaml.safe_load(f) or {}

    if not isinstance(raw_config, dict):
        raise ValueError(
            "Benchmark config file must contain a top-level mapping. "
            f"Got {type(raw_config).__name__}."
        )

    return BenchmarkConfig.create_from_dict(raw_config)


def main() -> None:
    config = _load_benchmark_config()

    os.makedirs(config.output_dir, exist_ok=True)
    with open(os.path.join(config.output_dir, "config.yaml"), "w") as f:
        yaml.dump(config.to_dict(), f)

    logger.info(f"Starting benchmark with config: {config}")

    set_seeds(config.seed)

    log_level = getattr(logging, config.log_level.upper())
    logging.basicConfig(
        format=LOGGER_FORMAT, level=log_level, datefmt=LOGGER_TIME_FORMAT
    )

    runner = BenchmarkRunnerLauncher(config)
    runner.run()


if __name__ == "__main__":
    main()
