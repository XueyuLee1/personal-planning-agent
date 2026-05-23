# Agentic Behavioral Workflow System

A rule-based prototype inspired by LLM-agent research on multi-turn workflows, reflective reasoning, and prompt optimisation. The system compares planned behaviour with actual behaviour, diagnoses common execution patterns, and produces an adaptive plan for the next session.

![UI screenshot](docs/ui-screenshot.png)

## Workflow Summary

- Observation layer extracts task counts, overlap signals, and distraction cues from raw text.
- Reasoning layer runs modular diagnostics for distraction, execution gaps, and planning mismatch.
- Aggregation layer combines diagnostic score deltas and records activated patterns.
- Reflection layer summarizes module-level reasoning and temporal meta-observations.
- Revision layer generates a next-session plan based on score thresholds and detected issues.
- Temporal memory stores scores and activated patterns in `history.json` for lightweight trend analysis across app restarts.

## Usage

Install dependencies if needed:

```bash
pip install gradio
```

Run the app:

```bash
python "app(2).py"
```

Open the local Gradio URL, enter planned behaviour and actual behaviour, then select **Analyze workflow**. The callback signature remains:

```python
behavioral_agent(planned_behavior: str, actual_behavior: str) -> str
```

Run the tests:

```bash
python -m unittest discover -s tests -v
```

## Persistent Memory

The app stores simple session history in `history.json`. Each record contains the session score and activated diagnostic patterns. This lets the report mention recurring issues without claiming that the system has learned a real policy.

To inspect history, open `history.json` in a text editor. To reset history, stop the app and delete `history.json`; the next run will start with empty memory. Missing, empty, or corrupted history files are handled gracefully.

## Modules

- `ObservationLayer`: gathers behavioural signals without scoring them.
- `DiagnosticResult`: bundles each module's pattern name, score delta, reasoning, reflection, and suggestion.
- `DistractionDetector`: identifies distracting tools, apps, or behaviours.
- `ExecutionGapAnalyzer`: estimates under-execution from planned versus actual task volume.
- `PlanningMismatchAnalyzer`: checks whether actual behaviour stayed aligned with the planned domain.
- `TaskSwitchAnalyzer`: detects possible task switching overload when the actual workflow contains more than three additional comma- or newline-separated task segments compared with the plan.
- `AggregationLayer`: computes the final score and activated patterns.
- `ReflectionLayer`: adds self-reflective and temporal observations.
- `RevisionLayer`: owns heuristic thresholds and next-session plan construction.

## Limitations

This system is heuristic and rule-based. It does not implement real LLM integration, reinforcement learning, retrieval-augmented memory, multi-agent collaboration, or meta-prompt optimisation. The text analysis uses simple lexical matching and task-count heuristics, so it should be treated as an interpretability demo rather than a clinical, educational, or productivity assessment.

## Future Directions

- Add learning components such as reinforcement learning or bandit-based prompt optimisation.
- Explore retrieval-augmented memory so older sessions can be queried more selectively.
- Add modular skill composition for richer workflow diagnostics.
- Add more nuanced cognitive state estimation for fatigue, uncertainty, affect, and task difficulty.
- Persist memory outside the Python process for longer-term trend analysis.
- Add selectable analysis focus modes for distractions, task management, and planning quality.
- Expand diagnostics into a collaborative multi-agent review loop.
