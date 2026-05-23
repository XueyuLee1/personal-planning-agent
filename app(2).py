"""Agentic Behavioral Workflow System.

This prototype keeps the public Gradio callback stable while organizing the
rule-based analysis into agent-inspired workflow layers.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import socket
from typing import Dict, Iterable, List

import gradio as gr

from temporal_memory import load_history, save_history


HISTORY_FILE = "history.json"
history_memory: List[Dict[str, object]] = []


@dataclass(frozen=True)
class Observation:
    planned_text: str
    actual_text: str
    planned_tasks: List[str]
    actual_tasks: List[str]
    planned_task_count: int
    actual_task_count: int
    distraction_hits: List[str]
    execution_ratio: float
    shared_terms_ratio: float


@dataclass(frozen=True)
class DiagnosticResult:
    pattern_name: str
    score_delta: int
    reasoning: str
    reflection: str
    suggestion: str


class ObservationLayer:
    """Collects behavioural signals without judging them."""

    DISTRACTION_TERMS = (
        "youtube",
        "tiktok",
        "instagram",
        "twitter",
        "x.com",
        "netflix",
        "game",
        "gaming",
        "scroll",
        "social media",
        "chat",
        "messages",
        "reddit",
    )

    @classmethod
    def collect(cls, planned_behavior: str, actual_behavior: str) -> Observation:
        planned_tasks = cls._extract_tasks(planned_behavior)
        actual_tasks = cls._extract_tasks(actual_behavior)
        planned_terms = cls._content_terms(planned_behavior)
        actual_terms = cls._content_terms(actual_behavior)
        shared_terms_ratio = (
            len(planned_terms & actual_terms) / max(1, len(planned_terms))
        )
        execution_ratio = len(actual_tasks) / max(1, len(planned_tasks))
        actual_lower = actual_behavior.lower()
        distraction_hits = [
            term for term in cls.DISTRACTION_TERMS if term in actual_lower
        ]

        return Observation(
            planned_text=planned_behavior.strip(),
            actual_text=actual_behavior.strip(),
            planned_tasks=planned_tasks,
            actual_tasks=actual_tasks,
            planned_task_count=len(planned_tasks),
            actual_task_count=len(actual_tasks),
            distraction_hits=distraction_hits,
            execution_ratio=execution_ratio,
            shared_terms_ratio=shared_terms_ratio,
        )

    @staticmethod
    def _extract_tasks(text: str) -> List[str]:
        normalized = text.replace("\n", ";")
        raw_items = []
        for chunk in normalized.split(";"):
            raw_items.extend(part.strip(" -0123456789.)\t") for part in chunk.split(","))
        return [item for item in raw_items if len(item.split()) >= 2]

    @staticmethod
    def _content_terms(text: str) -> set:
        stopwords = {
            "and",
            "the",
            "to",
            "of",
            "a",
            "an",
            "for",
            "in",
            "on",
            "with",
            "then",
            "after",
            "my",
            "i",
            "will",
            "did",
        }
        cleaned = "".join(
            char.lower() if char.isalnum() else " " for char in text
        )
        return {word for word in cleaned.split() if word not in stopwords and len(word) > 2}


class DistractionDetector:
    pattern_name = "DistractionDetector"

    def analyze(self, observation: Observation) -> DiagnosticResult:
        if not observation.distraction_hits:
            return DiagnosticResult(
                self.pattern_name,
                0,
                "No explicit distracting apps or behaviours were detected in the actual session.",
                "Attention appears reasonably protected in this report.",
                "Keep the current focus environment and preserve the same boundaries next time.",
            )

        hit_list = ", ".join(sorted(set(observation.distraction_hits)))
        penalty = -20 if len(observation.distraction_hits) >= 2 else -12
        return DiagnosticResult(
            self.pattern_name,
            penalty,
            f"Actual behaviour mentions distraction signals: {hit_list}.",
            "The session likely lost working memory bandwidth to competing cues.",
            "Block or remove the named apps before the next session and schedule a separate recovery break.",
        )


class ExecutionGapAnalyzer:
    pattern_name = "ExecutionGapAnalyzer"

    def analyze(self, observation: Observation) -> DiagnosticResult:
        gap = observation.planned_task_count - observation.actual_task_count
        if observation.execution_ratio >= 0.85:
            return DiagnosticResult(
                self.pattern_name,
                8,
                "Actual task volume closely matches the planned task volume.",
                "The plan seems executable at the current granularity.",
                "Repeat the same task load and add only one stretch item if energy remains high.",
            )
        if observation.execution_ratio >= 0.5:
            return DiagnosticResult(
                self.pattern_name,
                -12,
                f"Actual execution covered about {observation.execution_ratio:.0%} of the planned task count.",
                "There is a moderate execution gap, suggesting either time underestimation or friction during task switching.",
                "Reduce the next session by one task and define a visible finish line for the highest-value item.",
            )
        return DiagnosticResult(
            self.pattern_name,
            -28,
            f"Actual execution was far below the plan, leaving an estimated gap of {max(0, gap)} task(s).",
            "The plan may be over-estimating available capacity or under-estimating task difficulty.",
            "Choose one primary task, one tiny support task, and defer everything else.",
        )


class PlanningMismatchAnalyzer:
    pattern_name = "PlanningMismatchAnalyzer"

    def analyze(self, observation: Observation) -> DiagnosticResult:
        if observation.shared_terms_ratio >= 0.45:
            return DiagnosticResult(
                self.pattern_name,
                5,
                "Planned and actual descriptions share enough task language to indicate alignment.",
                "The user stayed near the intended work domain.",
                "Keep the plan wording concrete because it appears to anchor behaviour.",
            )
        if observation.shared_terms_ratio >= 0.2:
            return DiagnosticResult(
                self.pattern_name,
                -10,
                "The actual session only partially overlaps with the planned work.",
                "The plan may need clearer cues for what counts as success.",
                "Rewrite the next plan as observable actions instead of broad intentions.",
            )
        return DiagnosticResult(
            self.pattern_name,
            -22,
            "The actual session has little semantic overlap with the planned work.",
            "This looks like a planning mismatch rather than only an execution problem.",
            "Name the first concrete action and prepare materials before the session starts.",
        )


class TaskSwitchAnalyzer:
    pattern_name = "TaskSwitchAnalyzer: Task switching overload"

    @staticmethod
    def _split_workflow_tasks(text: str) -> List[str]:
        normalized = text.replace("\n", ",")
        return [segment.strip() for segment in normalized.split(",") if segment.strip()]

    def analyze(self, observation: Observation) -> DiagnosticResult:
        planned_tasks = self._split_workflow_tasks(observation.planned_text)
        actual_tasks = self._split_workflow_tasks(observation.actual_text)
        planned_count = len(planned_tasks)
        actual_count = len(actual_tasks)

        if actual_count > planned_count + 3:
            return DiagnosticResult(
                self.pattern_name,
                -5,
                (
                    "Observed more granular actual tasks than planned, "
                    f"with {actual_count} actual segments versus {planned_count} planned segments."
                ),
                "Multiple short tasks suggest fragmentation of attention and frequent context switching.",
                "Group similar tasks into batches and resist switching until the current batch is complete.",
            )

        return DiagnosticResult(
            self.pattern_name,
            0,
            "Actual task segmentation does not exceed the planned workflow by more than three tasks.",
            "Task switching does not appear to be the main source of efficiency loss in this session.",
            "Keep task boundaries visible and continue batching related work.",
        )


class AggregationLayer:
    BASE_SCORE = 100

    @classmethod
    def aggregate(cls, diagnostics: Iterable[DiagnosticResult]) -> Dict[str, object]:
        diagnostic_list = list(diagnostics)
        score = cls.BASE_SCORE + sum(item.score_delta for item in diagnostic_list)
        score = max(0, min(100, score))
        patterns = [
            item.pattern_name for item in diagnostic_list if item.score_delta < 0
        ]
        if not patterns:
            patterns = ["StableExecution"]
        return {
            "score": score,
            "patterns": patterns,
            "diagnostics": diagnostic_list,
        }


class ReflectionLayer:
    @staticmethod
    def summarize(
        observation: Observation,
        diagnostics: Iterable[DiagnosticResult],
        history: List[Dict[str, object]],
    ) -> str:
        diagnostic_list = list(diagnostics)
        reflections = [f"- [{item.pattern_name}] {item.reflection}" for item in diagnostic_list]
        if observation.execution_ratio < 0.5 and observation.planned_task_count >= 3:
            reflections.append(
                "- [MetaReflection] Repeated high task volume may signal capacity over-estimation."
            )
        if history:
            recent_scores = [int(record["score"]) for record in history[-3:]]
            avg_score = sum(recent_scores) / len(recent_scores)
            reflections.append(
                f"- [TemporalReflection] Recent average score is {avg_score:.1f}, so the next plan should adapt to trend rather than one isolated session."
            )
        return "\n".join(reflections)


class RevisionLayer:
    LOW_SCORE_THRESHOLD = 60
    MID_SCORE_THRESHOLD = 80

    @classmethod
    def revise(cls, score: int, patterns: List[str], observation: Observation) -> str:
        has_distraction = "DistractionDetector" in patterns
        if score < cls.LOW_SCORE_THRESHOLD:
            blockers = " Block distracting apps before starting." if has_distraction else ""
            return (
                "Next session plan: pick one essential task, make the first step visible, "
                f"and stop after a 25-minute focused block.{blockers}"
            )
        if score < cls.MID_SCORE_THRESHOLD:
            return (
                "Next session plan: keep the top two tasks, add a 5-minute setup ritual, "
                "and check progress halfway through."
            )
        if observation.planned_task_count <= 1:
            return (
                "Next session plan: repeat the focused structure and optionally add one small stretch task."
            )
        return (
            "Next session plan: preserve the current workload, keep success criteria explicit, "
            "and log any friction immediately after the session."
        )


def temporal_pattern_memory(history: List[Dict[str, object]]) -> str:
    if not history:
        return "No prior sessions stored yet. This run starts the temporal memory."

    pattern_counts: Dict[str, int] = {}
    for record in history:
        for pattern in record.get("patterns", []):
            pattern_counts[str(pattern)] = pattern_counts.get(str(pattern), 0) + 1

    recurring = [
        f"{pattern} appeared {count} time(s)"
        for pattern, count in sorted(pattern_counts.items())
        if count >= 2 and pattern != "StableExecution"
    ]
    latest_score = history[-1]["score"]
    trend_line = f"Stored sessions: {len(history)}. Latest previous score: {latest_score}."
    if recurring:
        return trend_line + "\nRecurring issues: " + "; ".join(recurring) + "."
    return trend_line + "\nNo recurring negative pattern has appeared twice yet."


def build_progress_bar(score: int) -> str:
    filled = int(round(score / 5))
    filled = max(0, min(20, filled))
    return "█" * filled + "─" * (20 - filled)


def build_report(
    session_id: str,
    score: int,
    patterns: List[str],
    diagnostics: List[DiagnosticResult],
    observation: Observation,
    history_before_update: List[Dict[str, object]],
) -> str:
    reasoning_lines = "\n".join(
        f"- [{item.pattern_name}] {item.reasoning}" for item in diagnostics
    )
    adaptation_lines = "\n".join(
        f"- [{item.pattern_name}] {item.suggestion}" for item in diagnostics
    )
    reflection_summary = ReflectionLayer.summarize(
        observation, diagnostics, history_before_update
    )
    revised_plan = RevisionLayer.revise(score, patterns, observation)
    progress_bar = build_progress_bar(score)

    return f"""# Agentic Behavioral Workflow Report

Session ID: `{session_id}`

## Current Productivity Score
{score} / 100  
{progress_bar}

## Activated Patterns
{", ".join(patterns)}

## Reason
{reasoning_lines}

## Reflect
{reflection_summary}

## Adapt
{adaptation_lines}

## Temporal Pattern Memory
{temporal_pattern_memory(history_before_update)}

## Revised Plan
{revised_plan}

## Next Step
Try the revised plan in your next work session, then return with the new actual behaviour so the workflow can compare sessions and adapt again.
"""


def run_workflow(planned_behavior: str, actual_behavior: str) -> str:
    observation = ObservationLayer.collect(planned_behavior, actual_behavior)
    analyzers = [
        DistractionDetector(),
        ExecutionGapAnalyzer(),
        PlanningMismatchAnalyzer(),
        TaskSwitchAnalyzer(),
    ]
    diagnostics = [analyzer.analyze(observation) for analyzer in analyzers]
    aggregation = AggregationLayer.aggregate(diagnostics)
    score = int(aggregation["score"])
    patterns = list(aggregation["patterns"])
    session_id = f"S{len(history_memory) + 1:03d}-{datetime.now().strftime('%Y%m%d%H%M%S')}"
    history_before_update = list(history_memory)
    report = build_report(
        session_id,
        score,
        patterns,
        list(aggregation["diagnostics"]),
        observation,
        history_before_update,
    )
    history_memory.append({"score": score, "patterns": patterns})
    save_history(HISTORY_FILE, history_memory)
    return report


def behavioral_agent(planned_behavior: str, actual_behavior: str) -> str:
    """Stable public callback used by the Gradio interface."""

    global history_memory
    history_memory = load_history(HISTORY_FILE)
    return run_workflow(planned_behavior, actual_behavior)


def load_css() -> str:
    with open("style.css", "r", encoding="utf-8") as css_file:
        return css_file.read()


def find_available_port(start: int = 7860, end: int = 8999) -> int:
    for port in range(start, end + 1):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
            probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                probe.bind(("127.0.0.1", port))
            except OSError:
                continue
            return port
    raise OSError(f"Cannot find an empty local port in range {start}-{end}.")


def create_demo() -> gr.Blocks:
    with gr.Blocks(title="Agentic Behavioral Workflow") as interface:
        gr.Markdown("# Agentic Behavioral Workflow System")
        gr.Markdown(
            "A rule-based prototype for reflective, multi-turn behavioural workflow analysis."
        )
        with gr.Row():
            planned_input = gr.Textbox(
                label="Planned behaviour",
                lines=8,
                placeholder="e.g. Study prompt optimisation and meta-learning; review Workflow-R1 notes; draft experiment plan",
            )
            actual_input = gr.Textbox(
                label="Actual behaviour",
                lines=8,
                placeholder="e.g. Reviewed one paper, checked messages, then lost time on YouTube before drafting notes",
            )
        analyze_button = gr.Button("Analyze workflow", variant="primary")
        output_report = gr.Markdown(label="Report")
        analyze_button.click(
            fn=behavioral_agent,
            inputs=[planned_input, actual_input],
            outputs=output_report,
        )
    return interface


demo = create_demo()


if __name__ == "__main__":
    demo.launch(css=load_css(), server_port=find_available_port())
