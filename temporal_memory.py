"""Small JSON-backed memory helpers for session history."""

from __future__ import annotations

import json
import os
from typing import List


def load_history(file_path: str) -> list:
    """Load session history from disk, falling back to an empty list.

    New session records contain structured planning fields, while older records
    may contain only score and patterns. Both formats are kept for backward
    compatibility.
    Corrupted or non-list files are treated as empty memory so the app can keep
    running during a demo.
    """

    if not os.path.exists(file_path) or os.path.getsize(file_path) == 0:
        return []

    try:
        with open(file_path, "r", encoding="utf-8") as history_file:
            history = json.load(history_file)
    except (json.JSONDecodeError, OSError):
        return []

    if not isinstance(history, list):
        return []

    valid_records = []
    for record in history:
        if not isinstance(record, dict):
            continue
        if "score" not in record or "patterns" not in record:
            continue
        valid_records.append(record)
    return valid_records


def save_history(file_path: str, history: List[dict]) -> None:
    """Persist session history as JSON, creating parent directories if needed."""

    directory = os.path.dirname(file_path)
    if directory:
        os.makedirs(directory, exist_ok=True)

    with open(file_path, "w", encoding="utf-8") as history_file:
        json.dump(history, history_file, indent=2)
