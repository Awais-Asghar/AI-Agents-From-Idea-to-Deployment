"""Entrypoint for running the Agentic AI workshop pipeline end-to-end."""
from __future__ import annotations

import argparse
import logging

from dotenv import load_dotenv

from crew import run_workshop_pipeline
from config.logging_config import configure_logging


def run_pipeline(topic: str) -> str:
    """Run the configured crew against the provided workshop topic."""
    load_dotenv()
    configure_logging()
    logging.getLogger(__name__).info("Starting workshop pipeline for topic: %s", topic)
    return run_workshop_pipeline(topic)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Agentic AI workshop crew pipeline.")
    parser.add_argument(
        "--topic",
        default="Agentic AI Workshop on Multi-Agent Systems",
        help="High-level theme to guide the crew's planning and content creation.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    output = run_pipeline(args.topic)
    print(output)
