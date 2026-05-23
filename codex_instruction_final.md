# Extended Improvement Instructions for the Agentic Behavioral Workflow System

These instructions build upon the previous refactor and will further enhance the `Agentic Behavioral Workflow System` to align more closely with state‑of‑the‑art LLM‑agent architectures.  Recent research shows that memory is a critical differentiator for autonomous agents – without persistent memory, an agent cannot learn from experience or avoid repeating mistakes【852264626031702†L94-L104】.  Modular design also improves flexibility and maintainability【331561216176033†L43-L62】.  You will implement a simple form of **long‑term memory**, add a new diagnostic module for task switching, and enrich the report with a visual progress bar.

## 1. Persist Temporal Memory Across Runs

Create a new module `temporal_memory.py` that reads and writes the session history to a JSON file.  It should expose two functions:

```python
def load_history(file_path: str) -> list:
    """Load an existing history file, returning a list of session dicts.
    If the file does not exist or is empty, return an empty list.
    Each record is expected to contain at least a `score` and `patterns` field."""

def save_history(file_path: str, history: list) -> None:
    """Persist the given history list to disk in JSON format.  Ensure the directory
    exists and create or overwrite the file."""
```

In `app(2).py`, set a constant `HISTORY_FILE = "history.json"` and, at the start of `behavioral_agent`, load the history via `load_history(HISTORY_FILE)` into `history_memory`.  After computing the current session’s score and patterns, append `{'score': score, 'patterns': patterns}` to this list, then call `save_history(HISTORY_FILE, history_memory)` to persist the update.  This ensures that historical data survives restarts and enables analysis of recurring patterns, echoing the idea that memory transforms a stateless model into an adaptive agent【852264626031702†L94-L103】.

## 2. Add a TaskSwitchAnalyzer Diagnostic

Introduce a new diagnostic module named `TaskSwitchAnalyzer` that detects excessive task switching.  Frequent task switching is a well‑known productivity killer and fits within Dai’s framework of behavioural diagnostics.  The module should:

* Accept the planned and actual workflow texts.
* Compute `planned_tasks` and `actual_tasks` by splitting each text on commas (`,`) or newline (`\n`), trimming whitespace, and counting non‑empty segments.
* If `actual_tasks` exceeds `planned_tasks + 3`, consider it evidence of over‑switching.
* Return a `DiagnosticResult` with a `pattern_name` like `[TaskSwitchAnalyzer] Task switching overload`, a `score_delta` (e.g. –5), a reasoning string (e.g. "Observed more granular tasks than planned, indicating frequent context switching"), a reflection (e.g. "Multiple short tasks suggest fragmentation of attention"), and an adaptation suggestion (e.g. "Group similar tasks into batches and resist switching until completion").

Register `TaskSwitchAnalyzer` alongside the existing diagnostic modules and include its output when aggregating results.  Adjust the overall score by summing all `score_delta` values from active diagnostics, ensuring that scores remain bounded between 0 and 100.  Update the operator analysis section of the report to include the new pattern when triggered.

## 3. Add a Progress Bar to the Report

To improve interpretability, embed a simple progress bar that visualises the current productivity score.  In the report generation section, after computing `score`, create a string `progress_bar` consisting of 20 segments: multiply the score (0–100) by 0.2, then round to the nearest integer `n_bars`.  Use `█` (U+2588) for filled segments and `─` (U+2500) for empty ones:

```python
filled = int(round(score / 5))  # 20 segments correspond to 100 points
progress_bar = '█' * filled + '─' * (20 - filled)
```

Include this progress bar under **Current Productivity Score** in the report, e.g.:

```
### Current Productivity Score
85 / 100  
█████████████──────
```

This visual cue helps users quickly assess performance trends.

## 4. Extend Unit Tests

Add new tests in `tests/test_diagnostics.py` to cover:

1. The `TaskSwitchAnalyzer`: ensure that when `actual_tasks` > `planned_tasks + 3`, it returns a negative score delta.
2. The persistent history: simulate two runs by manually invoking the memory read/write functions and verify that both sessions appear in the loaded history.  You can mock file paths with `tempfile.NamedTemporaryFile()`.
3. The progress bar: given a specific score (e.g. 50), assert that the progress bar contains the correct number of filled and empty segments.

These tests will maintain the reliability of your refactored system, consistent with unit testing best practices【227081749432307†L80-L87】.

## 5. Update Documentation and Requirements

* **README(1).md**: Document the new persistent memory mechanism, explain that history is stored in a JSON file, and describe how users can reset or inspect their history.  Add a subsection for **TaskSwitchAnalyzer**, explaining its purpose and heuristics.  Update the Limitations section to emphasise that this is still a rule‑based prototype without real LLM integration or learning, and list future directions such as integrating retrieval‑augmented memory【852264626031702†L94-L104】 and modular skill composition【331561216176033†L51-L54】.
* **requirements(1).txt**: Add the `json` module is built‑in, so no dependency changes are needed.

## 6. Ensure Backward Compatibility

Preserve the function signature of `behavioral_agent(planned, actual)` so that the Gradio interface remains unchanged.  When reading the history file, handle missing or corrupted files gracefully and fall back to an empty list.

After implementing these changes, run `python -m unittest discover -s tests -v` to ensure all tests pass and launch the Gradio app to verify the report shows the progress bar and new diagnostics.  Commit your changes with a descriptive message.
