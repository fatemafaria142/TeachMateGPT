"""
Structured logger for agent calls and system events.
"""
from __future__ import annotations

import json
import logging
import os
from datetime import datetime
from pathlib import Path

import config

Path(config.LOGS_DIR).mkdir(parents=True, exist_ok=True)

_log_file = os.path.join(
    config.LOGS_DIR,
    datetime.now().strftime("session_%Y%m%d_%H%M%S.log"),
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler(_log_file, encoding="utf-8"),
        logging.StreamHandler(),
    ],
)

_logger = logging.getLogger("bangla_tutor")


def log_agent(agent_name: str, input_text: str, output_text: str, **extra) -> None:
    """Log an agent call with input/output."""
    record = {
        "agent":  agent_name,
        "input":  input_text[:300],
        "output": output_text[:300],
        **extra,
    }
    _logger.info("AGENT | %s", json.dumps(record, ensure_ascii=False))


def log_retrieval(query: str, num_results: int, chapters: list[str]) -> None:
    """Log a retrieval event."""
    _logger.info(
        "RETRIEVAL | query=%s | results=%d | chapters=%s",
        query[:100],
        num_results,
        chapters,
    )


def log_error(context: str, error: Exception) -> None:
    """Log an error."""
    _logger.error("ERROR | context=%s | error=%s", context, str(error))


def log_info(message: str) -> None:
    """Log a plain info message."""
    _logger.info("INFO | %s", message)


def log_warning(message: str) -> None:
    """Log a warning message."""
    _logger.warning("WARN | %s", message)
