"""Small JSON-backed memory helpers for session history."""

from __future__ import annotations

import json
import os
from typing import List


def load_history(file_path: str) -> list:
    """Load session history from disk, falling back to an empty list.

    Each valid record is expected to contain at least a score and patterns field.
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

    return [
        record
        for record in history
        if isinstance(record, dict)
        and "score" in record
        and "patterns" in record
    ]


def save_history(file_path: str, history: List[dict]) -> None:
    """Persist session history as JSON, creating parent directories if needed."""

    directory = os.path.dirname(file_path)
    if directory:
        os.makedirs(directory, exist_ok=True)

    with open(file_path, "w", encoding="utf-8") as history_file:
        json.dump(history, history_file, indent=2)
