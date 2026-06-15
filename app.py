"""Personal Planning Agent.

This prototype keeps the public Gradio callback stable while organizing the
rule-based analysis into agent-inspired workflow layers.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import socket
from typing import Dict, Iterable, List, Optional

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


@dataclass(frozen=True)
class SessionRecord:
    session_id: str
    created_at: str
    planned_text: str
    actual_text: str
    planned_tasks: List[str]
    actual_tasks: List[str]
    planned_task_count: int
    actual_task_count: int
    score: int
    patterns: List[str]

    def to_dict(self) -> Dict[str, object]:
        return {
            "session_id": self.session_id,
            "created_at": self.created_at,
            "planned_text": self.planned_text,
            "actual_text": self.actual_text,
            "planned_tasks": self.planned_tasks,
            "actual_tasks": self.actual_tasks,
            "planned_task_count": self.planned_task_count,
            "actual_task_count": self.actual_task_count,
            "score": self.score,
            "patterns": self.patterns,
        }


@dataclass(frozen=True)
class UserProfile:
    total_sessions: int
    average_score: Optional[float]
    average_planned_tasks: Optional[float]
    average_actual_tasks: Optional[float]
    average_completion_rate: Optional[float]
    overplanning_frequency: Optional[float]
    common_patterns: List[str]
    recent_score_trend: str


@dataclass(frozen=True)
class CapacityEstimate:
    estimated_task_capacity: Optional[int]
    reliable_task_range: str
    confidence: str
    basis: str
    risk_notes: List[str]


@dataclass(frozen=True)
class PlanningRecommendation:
    recommended_task_count: int
    recommended_tasks: List[str]
    buffer_policy: str
    success_probability: int
    rationale: str
    risk_warnings: List[str]


@dataclass(frozen=True)
class CandidateTask:
    task_name: str
    estimated_minutes: int
    importance: int
    urgency: int


@dataclass(frozen=True)
class ScoredTask:
    task: CandidateTask
    priority_score: float
    feasibility_score: float
    final_score: float


@dataclass(frozen=True)
class TaskSelectionResult:
    selected_tasks: List[CandidateTask]
    deferred_tasks: List[CandidateTask]


@dataclass(frozen=True)
class TaskPlanItem:
    task_name: str
    allocated_minutes: int
    order: int
    reason: str
    is_partial: bool


@dataclass(frozen=True)
class DeferredTask:
    task_name: str
    estimated_minutes: int
    defer_type: str
    reason: str


@dataclass(frozen=True)
class TaskLevelPlan:
    available_minutes: int
    work_minutes: int
    buffer_minutes: int
    selected_tasks: List[TaskPlanItem]
    deferred_tasks: List[DeferredTask]
    planning_risks: List[str]
    plan_confidence: str


@dataclass(frozen=True)
class PlanOutcomeRecord:
    plan_id: str
    created_at: str
    planned_tasks: List[str]
    planned_minutes: int
    completed_tasks: List[str]
    actual_minutes: int
    interruption_count: int
    task_switch_count: int
    fatigue_level: int
    completion_rate: float
    notes: str

    def to_dict(self) -> Dict[str, object]:
        return {
            "record_type": "plan_outcome",
            "plan_id": self.plan_id,
            "created_at": self.created_at,
            "planned_tasks": self.planned_tasks,
            "planned_minutes": self.planned_minutes,
            "completed_tasks": self.completed_tasks,
            "actual_minutes": self.actual_minutes,
            "interruption_count": self.interruption_count,
            "task_switch_count": self.task_switch_count,
            "fatigue_level": self.fatigue_level,
            "completion_rate": self.completion_rate,
            "notes": self.notes,
            "score": int(round(self.completion_rate * 100)),
            "patterns": ["PlanOutcomeRecord"],
        }


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


def _mean(values: List[float]) -> Optional[float]:
    if not values:
        return None
    return sum(values) / len(values)


def _format_optional_number(value: Optional[float], suffix: str = "") -> str:
    if value is None:
        return "not enough data"
    return f"{value:.2f}{suffix}"


def build_user_profile(history: List[Dict[str, object]]) -> UserProfile:
    scores: List[float] = []
    planned_counts: List[float] = []
    actual_counts: List[float] = []
    completion_rates: List[float] = []
    overplanned_sessions = 0
    task_count_sessions = 0
    pattern_counts: Dict[str, int] = {}

    for record in history:
        score = record.get("score")
        if isinstance(score, (int, float)):
            scores.append(float(score))

        patterns = record.get("patterns", [])
        if isinstance(patterns, list):
            for pattern in patterns:
                pattern_name = str(pattern)
                pattern_counts[pattern_name] = pattern_counts.get(pattern_name, 0) + 1

        planned_count = record.get("planned_task_count")
        actual_count = record.get("actual_task_count")
        if isinstance(planned_count, (int, float)) and isinstance(actual_count, (int, float)):
            if planned_count > 0:
                planned_counts.append(float(planned_count))
                actual_counts.append(float(actual_count))
                completion_rates.append(min(float(actual_count) / float(planned_count), 1.0))
                task_count_sessions += 1
                if actual_count < planned_count:
                    overplanned_sessions += 1

    common_patterns = [
        pattern
        for pattern, _count in sorted(
            pattern_counts.items(), key=lambda item: (-item[1], item[0])
        )[:3]
    ]

    recent_score_trend = "not enough data"
    if len(scores) >= 4:
        previous_avg = sum(scores[-4:-2]) / 2
        recent_avg = sum(scores[-2:]) / 2
        if recent_avg > previous_avg + 3:
            recent_score_trend = "improving"
        elif recent_avg < previous_avg - 3:
            recent_score_trend = "declining"
        else:
            recent_score_trend = "stable"

    overplanning_frequency = None
    if task_count_sessions:
        overplanning_frequency = overplanned_sessions / task_count_sessions

    return UserProfile(
        total_sessions=len(history),
        average_score=_mean(scores),
        average_planned_tasks=_mean(planned_counts),
        average_actual_tasks=_mean(actual_counts),
        average_completion_rate=_mean(completion_rates),
        overplanning_frequency=overplanning_frequency,
        common_patterns=common_patterns,
        recent_score_trend=recent_score_trend,
    )


def format_user_profile(profile: UserProfile) -> str:
    if profile.total_sessions == 0:
        return "No historical sessions are available yet. The profile will become more useful after several saved sessions."

    pattern_text = ", ".join(profile.common_patterns) if profile.common_patterns else "none yet"
    completion_rate = (
        "not enough data"
        if profile.average_completion_rate is None
        else f"{profile.average_completion_rate:.0%}"
    )
    overplanning = (
        "not enough data"
        if profile.overplanning_frequency is None
        else f"{profile.overplanning_frequency:.0%}"
    )

    return "\n".join(
        [
            f"- Total recorded sessions: {profile.total_sessions}",
            f"- Average score: {_format_optional_number(profile.average_score)}",
            f"- Average planned tasks: {_format_optional_number(profile.average_planned_tasks)}",
            f"- Average actual tasks: {_format_optional_number(profile.average_actual_tasks)}",
            f"- Average task completion rate: {completion_rate}",
            f"- Overplanning frequency: {overplanning}",
            f"- Common activated patterns: {pattern_text}",
            f"- Recent score trend: {profile.recent_score_trend}",
        ]
    )


class CapacityEstimationEngine:
    """Estimates task-count capacity from historical behaviour records."""

    @staticmethod
    def estimate(profile: UserProfile, observation: Observation) -> CapacityEstimate:
        if profile.average_actual_tasks is None:
            return CapacityEstimate(
                estimated_task_capacity=None,
                reliable_task_range="not enough data",
                confidence="low",
                basis="No structured task history is available yet.",
                risk_notes=[
                    "Use the current plan conservatively until several sessions have been recorded."
                ],
            )

        base_capacity = max(1, int(round(profile.average_actual_tasks)))
        lower_bound = max(1, base_capacity - 1)
        upper_bound = max(lower_bound, base_capacity)
        risk_notes: List[str] = []

        if profile.overplanning_frequency is not None and profile.overplanning_frequency >= 0.6:
            upper_bound = max(lower_bound, upper_bound - 1)
            risk_notes.append(
                "History shows frequent overplanning, so the reliable upper bound is reduced."
            )

        if observation.planned_task_count > upper_bound:
            risk_notes.append(
                f"Current plan has {observation.planned_task_count} tasks, above the estimated reliable range."
            )

        if profile.average_completion_rate is not None and profile.average_completion_rate < 0.6:
            risk_notes.append(
                "Average task completion rate is below 60%, so future plans should stay smaller."
            )

        confidence = "medium"
        if profile.total_sessions >= 5 and profile.average_completion_rate is not None:
            confidence = "high"
        elif profile.total_sessions < 3:
            confidence = "low"

        basis = (
            "Estimated from historical actual task counts"
            f" (average actual tasks: {profile.average_actual_tasks:.2f})."
        )

        return CapacityEstimate(
            estimated_task_capacity=base_capacity,
            reliable_task_range=f"{lower_bound}-{upper_bound} tasks",
            confidence=confidence,
            basis=basis,
            risk_notes=risk_notes or ["No major task-count capacity risk detected."],
        )


def format_capacity_estimate(estimate: CapacityEstimate) -> str:
    capacity_text = (
        "not enough data"
        if estimate.estimated_task_capacity is None
        else f"{estimate.estimated_task_capacity} task(s)"
    )
    risks = "\n".join(f"- {note}" for note in estimate.risk_notes)
    return "\n".join(
        [
            f"- Estimated task capacity: {capacity_text}",
            f"- Reliable task range: {estimate.reliable_task_range}",
            f"- Confidence: {estimate.confidence}",
            f"- Basis: {estimate.basis}",
            "- Risk notes:",
            risks,
        ]
    )


class PlanningRecommendationEngine:
    """Creates a conservative next-plan recommendation from profile and capacity."""

    @staticmethod
    def recommend(
        observation: Observation,
        profile: UserProfile,
        capacity_estimate: CapacityEstimate,
    ) -> PlanningRecommendation:
        if capacity_estimate.estimated_task_capacity is None:
            recommended_count = min(max(1, observation.planned_task_count), 2)
            confidence_penalty = 15
            rationale = (
                "Not enough historical task data is available, so the system recommends a small starter plan."
            )
        else:
            recommended_count = min(
                max(1, capacity_estimate.estimated_task_capacity),
                max(1, observation.planned_task_count),
            )
            confidence_penalty = 0 if capacity_estimate.confidence == "high" else 5
            rationale = (
                "Recommended task count is capped by the user's historical actual task capacity."
            )

        if profile.overplanning_frequency is not None and profile.overplanning_frequency >= 0.6:
            recommended_count = max(1, recommended_count - 1)
            rationale += " Frequent overplanning reduces the recommended workload by one task."

        recommended_tasks = observation.planned_tasks[:recommended_count]
        if not recommended_tasks and observation.planned_text:
            recommended_tasks = [observation.planned_text]

        base_probability = 65
        if profile.average_completion_rate is not None:
            base_probability = int(round(profile.average_completion_rate * 100))

        probability = base_probability - confidence_penalty
        risk_warnings: List[str] = []

        if observation.planned_task_count > recommended_count:
            probability -= 10
            risk_warnings.append(
                f"Original plan has {observation.planned_task_count} tasks; recommendation limits it to {recommended_count}."
            )

        if profile.overplanning_frequency is not None and profile.overplanning_frequency >= 0.6:
            probability -= 10
            risk_warnings.append(
                "Historical overplanning frequency is high, so adding extra tasks is risky."
            )

        if profile.recent_score_trend == "declining":
            probability -= 5
            risk_warnings.append(
                "Recent score trend is declining; keep the next plan especially small."
            )
        elif profile.recent_score_trend == "improving":
            probability += 5

        if capacity_estimate.confidence == "low":
            risk_warnings.append(
                "Capacity estimate confidence is low because there are not enough structured sessions."
            )

        success_probability = max(10, min(95, probability))
        buffer_policy = "Reserve one task-sized buffer; do not add replacement tasks if you finish early."
        if recommended_count == 1:
            buffer_policy = "Focus on one primary task and keep the rest of the session as buffer."

        return PlanningRecommendation(
            recommended_task_count=recommended_count,
            recommended_tasks=recommended_tasks,
            buffer_policy=buffer_policy,
            success_probability=success_probability,
            rationale=rationale,
            risk_warnings=risk_warnings or ["No major recommendation risk detected."],
        )


class TaskPlanningEngine:
    """Scores and selects candidate tasks for a concrete planning session."""

    DEFAULT_TASK_LIMIT = 2
    VALID_DEFER_TYPES = {"capacity_limit", "time_limit", "low_priority"}

    @staticmethod
    def calculate_buffer(available_minutes: int) -> Dict[str, int]:
        if available_minutes <= 0:
            return {"buffer_minutes": 0, "work_minutes": 0}
        buffer_minutes = max(10, round(available_minutes * 0.15))
        buffer_minutes = min(buffer_minutes, available_minutes)
        return {
            "buffer_minutes": buffer_minutes,
            "work_minutes": available_minutes - buffer_minutes,
        }

    @staticmethod
    def score_task(task: CandidateTask, available_minutes: int) -> ScoredTask:
        priority_score = task.importance * 0.6 + task.urgency * 0.4
        safe_estimate = max(1, task.estimated_minutes)
        feasibility_score = min(1.0, max(0, available_minutes) / safe_estimate)
        normalized_priority = priority_score / 5
        final_score = normalized_priority * 0.7 + feasibility_score * 0.3
        return ScoredTask(
            task=task,
            priority_score=priority_score,
            feasibility_score=feasibility_score,
            final_score=final_score,
        )

    @classmethod
    def score_tasks(
        cls,
        candidate_tasks: List[CandidateTask],
        available_minutes: int,
    ) -> List[ScoredTask]:
        return [
            cls.score_task(task, available_minutes)
            for task in candidate_tasks
            if task.estimated_minutes > 0
        ]

    @classmethod
    def select_tasks(
        cls,
        available_minutes: int,
        candidate_tasks: List[CandidateTask],
        capacity_estimate: CapacityEstimate,
    ) -> TaskSelectionResult:
        scored_tasks = cls.score_tasks(candidate_tasks, available_minutes)
        sorted_tasks = sorted(
            scored_tasks,
            key=lambda item: (
                item.final_score,
                item.task.urgency,
                item.task.importance,
                -item.task.estimated_minutes,
            ),
            reverse=True,
        )
        task_limit = capacity_estimate.estimated_task_capacity or cls.DEFAULT_TASK_LIMIT
        task_limit = max(1, task_limit)
        remaining_minutes = max(0, available_minutes)
        selected: List[CandidateTask] = []
        deferred: List[CandidateTask] = []

        for scored_task in sorted_tasks:
            task = scored_task.task
            if (
                len(selected) < task_limit
                and task.estimated_minutes <= remaining_minutes
            ):
                selected.append(task)
                remaining_minutes -= task.estimated_minutes
            else:
                deferred.append(task)

        selected_names = {task.task_name for task in selected}
        deferred_names = {task.task_name for task in deferred}
        for task in candidate_tasks:
            if task.task_name not in selected_names and task.task_name not in deferred_names:
                deferred.append(task)

        return TaskSelectionResult(
            selected_tasks=selected,
            deferred_tasks=deferred,
        )

    @staticmethod
    def order_tasks(selected_tasks: List[CandidateTask]) -> List[CandidateTask]:
        return sorted(
            selected_tasks,
            key=lambda task: (-task.urgency, -task.importance, task.estimated_minutes),
        )

    @staticmethod
    def allocate_time(
        ordered_tasks: List[CandidateTask],
        work_minutes: int,
    ) -> List[TaskPlanItem]:
        remaining_minutes = max(0, work_minutes)
        plan_items: List[TaskPlanItem] = []

        for index, task in enumerate(ordered_tasks, start=1):
            allocated_minutes = min(task.estimated_minutes, remaining_minutes)
            is_partial = allocated_minutes < task.estimated_minutes
            reason = (
                "Partial progress only because the task exceeds remaining work time."
                if is_partial
                else "Selected because it fits the work budget and has high priority."
            )
            plan_items.append(
                TaskPlanItem(
                    task_name=task.task_name,
                    allocated_minutes=allocated_minutes,
                    order=index,
                    reason=reason,
                    is_partial=is_partial,
                )
            )
            remaining_minutes -= allocated_minutes

        return plan_items

    @classmethod
    def explain_deferred_tasks(
        cls,
        deferred_tasks: List[CandidateTask],
        selected_tasks: List[CandidateTask],
        work_minutes: int,
        capacity_estimate: CapacityEstimate,
    ) -> List[DeferredTask]:
        task_limit = capacity_estimate.estimated_task_capacity or cls.DEFAULT_TASK_LIMIT
        task_limit = max(1, task_limit)
        selected_minutes = sum(task.estimated_minutes for task in selected_tasks)
        remaining_minutes = max(0, work_minutes - selected_minutes)
        explained: List[DeferredTask] = []

        for task in deferred_tasks:
            if task.estimated_minutes > work_minutes:
                defer_type = "time_limit"
                reason = "Deferred because estimated duration exceeds today's work budget."
            elif len(selected_tasks) >= task_limit:
                defer_type = "capacity_limit"
                reason = "Deferred because historical capacity suggests limiting task count."
            elif task.estimated_minutes > remaining_minutes:
                defer_type = "time_limit"
                reason = "Deferred because higher-priority tasks consumed available work time."
            else:
                defer_type = "low_priority"
                reason = "Deferred because it was lower priority for this session."

            explained.append(
                DeferredTask(
                    task_name=task.task_name,
                    estimated_minutes=task.estimated_minutes,
                    defer_type=defer_type,
                    reason=reason,
                )
            )

        return explained

    @staticmethod
    def _build_planning_risks(
        candidate_tasks: List[CandidateTask],
        selected_items: List[TaskPlanItem],
        deferred_tasks: List[DeferredTask],
        work_minutes: int,
        capacity_estimate: CapacityEstimate,
    ) -> List[str]:
        risks: List[str] = []
        if not candidate_tasks:
            risks.append("No candidate tasks were provided.")
        if work_minutes <= 0:
            risks.append("No work time remains after protecting buffer time.")
        if capacity_estimate.estimated_task_capacity is None:
            risks.append("No historical capacity data is available yet.")
        if deferred_tasks:
            risks.append("Some candidate tasks were deferred due to time, capacity, or priority limits.")
        if any(item.is_partial for item in selected_items):
            risks.append("At least one selected task is scheduled as partial progress only.")

        allocated_minutes = sum(item.allocated_minutes for item in selected_items)
        if work_minutes > 0 and allocated_minutes >= round(work_minutes * 0.9):
            risks.append("Selected workload is close to the available work budget.")

        return risks or ["No major planning risks detected."]

    @staticmethod
    def _derive_plan_confidence(
        selected_items: List[TaskPlanItem],
        deferred_tasks: List[DeferredTask],
        capacity_estimate: CapacityEstimate,
        planning_risks: List[str],
    ) -> str:
        if not selected_items or capacity_estimate.confidence == "low":
            return "low"
        meaningful_risks = [
            risk for risk in planning_risks if risk != "No major planning risks detected."
        ]
        if capacity_estimate.confidence == "high" and not deferred_tasks and not meaningful_risks:
            return "high"
        return "medium"

    @classmethod
    def generate_task_level_plan(
        cls,
        available_minutes: int,
        candidate_tasks: List[CandidateTask],
        capacity_estimate: CapacityEstimate,
    ) -> TaskLevelPlan:
        buffer_info = cls.calculate_buffer(available_minutes)
        work_minutes = buffer_info["work_minutes"]
        buffer_minutes = buffer_info["buffer_minutes"]
        selection = cls.select_tasks(work_minutes, candidate_tasks, capacity_estimate)
        ordered_tasks = cls.order_tasks(selection.selected_tasks)
        selected_items = cls.allocate_time(ordered_tasks, work_minutes)
        deferred_tasks = cls.explain_deferred_tasks(
            selection.deferred_tasks,
            selection.selected_tasks,
            work_minutes,
            capacity_estimate,
        )
        planning_risks = cls._build_planning_risks(
            candidate_tasks,
            selected_items,
            deferred_tasks,
            work_minutes,
            capacity_estimate,
        )
        plan_confidence = cls._derive_plan_confidence(
            selected_items,
            deferred_tasks,
            capacity_estimate,
            planning_risks,
        )

        return TaskLevelPlan(
            available_minutes=max(0, available_minutes),
            work_minutes=work_minutes,
            buffer_minutes=buffer_minutes,
            selected_tasks=selected_items,
            deferred_tasks=deferred_tasks,
            planning_risks=planning_risks,
            plan_confidence=plan_confidence,
        )


def format_planning_recommendation(recommendation: PlanningRecommendation) -> str:
    tasks = "\n".join(
        f"{index}. {task}"
        for index, task in enumerate(recommendation.recommended_tasks, start=1)
    )
    risks = "\n".join(f"- {warning}" for warning in recommendation.risk_warnings)
    return "\n".join(
        [
            f"- Recommended task count: {recommendation.recommended_task_count}",
            "- Recommended tasks:",
            tasks if tasks else "No task text available.",
            f"- Buffer policy: {recommendation.buffer_policy}",
            f"- Estimated success probability: {recommendation.success_probability}%",
            f"- Rationale: {recommendation.rationale}",
            "- Risk warnings:",
            risks,
        ]
    )


def generate_plan_report(task_level_plan: TaskLevelPlan) -> str:
    selected_tasks = "\n".join(
        (
            f"{item.order}. {item.task_name} - {item.allocated_minutes} min"
            f"{' (partial)' if item.is_partial else ''}\n"
            f"   Reason: {item.reason}"
        )
        for item in task_level_plan.selected_tasks
    )
    deferred_tasks = "\n".join(
        (
            f"- {task.task_name} ({task.estimated_minutes} min)\n"
            f"  Type: {task.defer_type}\n"
            f"  Reason: {task.reason}"
        )
        for task in task_level_plan.deferred_tasks
    )
    task_order = "\n".join(
        f"{item.order}. {item.task_name}" for item in task_level_plan.selected_tasks
    )
    planning_risks = "\n".join(
        f"- {risk}" for risk in task_level_plan.planning_risks
    )

    return f"""# Personal Planning Report

## Available Time
{task_level_plan.available_minutes} minutes

## Work Time
{task_level_plan.work_minutes} minutes

## Protected Buffer
{task_level_plan.buffer_minutes} minutes

## Selected Tasks
{selected_tasks if selected_tasks else "No tasks selected."}

## Deferred Tasks
{deferred_tasks if deferred_tasks else "No tasks deferred."}

## Task Order
{task_order if task_order else "No task order available."}

## Plan Confidence
{task_level_plan.plan_confidence}

## Planning Risks
{planning_risks}
"""


def calculate_completion_rate(planned_tasks: List[str], completed_tasks: List[str]) -> float:
    if not planned_tasks:
        return 0.0
    normalized_completed = {
        task.strip().lower() for task in completed_tasks if task.strip()
    }
    completed_count = sum(
        1 for task in planned_tasks if task.strip().lower() in normalized_completed
    )
    return min(1.0, completed_count / len(planned_tasks))


def parse_completed_tasks(completed_tasks_text: str) -> List[str]:
    if not completed_tasks_text:
        return []
    normalized = completed_tasks_text.replace("\n", ",").replace(";", ",")
    return [task.strip() for task in normalized.split(",") if task.strip()]


def build_plan_state(task_level_plan: TaskLevelPlan) -> Dict[str, object]:
    return {
        "plan_id": f"P{datetime.now().strftime('%Y%m%d%H%M%S')}",
        "planned_tasks": [task.task_name for task in task_level_plan.selected_tasks],
        "planned_minutes": sum(task.allocated_minutes for task in task_level_plan.selected_tasks),
    }


def build_plan_outcome_record(
    plan_state: Dict[str, object],
    completed_tasks: List[str],
    actual_minutes: int,
    interruption_count: int,
    task_switch_count: int,
    fatigue_level: int,
    notes: str,
) -> PlanOutcomeRecord:
    planned_tasks = [
        str(task) for task in plan_state.get("planned_tasks", []) if str(task).strip()
    ]
    completion_rate = calculate_completion_rate(planned_tasks, completed_tasks)
    return PlanOutcomeRecord(
        plan_id=str(plan_state.get("plan_id", f"P{datetime.now().strftime('%Y%m%d%H%M%S')}")),
        created_at=datetime.now().isoformat(timespec="seconds"),
        planned_tasks=planned_tasks,
        planned_minutes=max(0, _coerce_int(plan_state.get("planned_minutes"))),
        completed_tasks=completed_tasks,
        actual_minutes=max(0, actual_minutes),
        interruption_count=max(0, interruption_count),
        task_switch_count=max(0, task_switch_count),
        fatigue_level=max(1, min(5, fatigue_level or 1)),
        completion_rate=completion_rate,
        notes=notes.strip() if notes else "",
    )


def generate_outcome_summary(outcome: PlanOutcomeRecord) -> str:
    observations: List[str] = []
    if outcome.completion_rate >= 0.8:
        observations.append("Most planned tasks were completed.")
    elif outcome.completion_rate >= 0.5:
        observations.append("The plan was partially completed.")
    else:
        observations.append("The plan was mostly unfinished.")

    if outcome.actual_minutes < outcome.planned_minutes:
        observations.append("Actual study time was lower than planned time.")
    elif outcome.actual_minutes > outcome.planned_minutes:
        observations.append("Actual study time exceeded planned time.")
    else:
        observations.append("Actual study time matched planned time.")

    if outcome.interruption_count >= 3:
        observations.append("Interruptions were high enough to threaten focus.")
    if outcome.task_switch_count >= 5:
        observations.append("Task switching was high and may have fragmented attention.")
    if outcome.fatigue_level >= 4:
        observations.append("Fatigue was high; future plans should preserve more buffer.")

    return "\n".join(
        [
            f"- Plan completion rate: {outcome.completion_rate:.0%}",
            "- Execution summary:",
            *[f"  - {observation}" for observation in observations],
        ]
    )


def generate_plan_vs_outcome_report(outcome: PlanOutcomeRecord) -> str:
    planned_tasks = "\n".join(f"- {task}" for task in outcome.planned_tasks)
    completed_tasks = "\n".join(f"- {task}" for task in outcome.completed_tasks)
    return f"""# Plan vs Outcome Report

## Planned Tasks
{planned_tasks if planned_tasks else "No planned tasks recorded."}

## Completed Tasks
{completed_tasks if completed_tasks else "No completed tasks recorded."}

## Planned Minutes
{outcome.planned_minutes} minutes

## Actual Minutes
{outcome.actual_minutes} minutes

## Completion Rate
{outcome.completion_rate:.0%}

## Interruptions
{outcome.interruption_count}

## Task Switches
{outcome.task_switch_count}

## Fatigue
{outcome.fatigue_level} / 5

## Outcome Summary
{generate_outcome_summary(outcome)}
"""


def build_progress_bar(score: int) -> str:
    filled = int(round(score / 5))
    filled = max(0, min(20, filled))
    return "█" * filled + "─" * (20 - filled)


def build_session_record(
    session_id: str,
    observation: Observation,
    score: int,
    patterns: List[str],
) -> SessionRecord:
    return SessionRecord(
        session_id=session_id,
        created_at=datetime.now().isoformat(timespec="seconds"),
        planned_text=observation.planned_text,
        actual_text=observation.actual_text,
        planned_tasks=observation.planned_tasks,
        actual_tasks=observation.actual_tasks,
        planned_task_count=observation.planned_task_count,
        actual_task_count=observation.actual_task_count,
        score=score,
        patterns=patterns,
    )


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
    user_profile = build_user_profile(history_before_update)
    user_profile_summary = format_user_profile(user_profile)
    capacity_estimate = CapacityEstimationEngine.estimate(user_profile, observation)
    capacity_summary = format_capacity_estimate(capacity_estimate)
    planning_recommendation = PlanningRecommendationEngine.recommend(
        observation, user_profile, capacity_estimate
    )
    planning_summary = format_planning_recommendation(planning_recommendation)

    return f"""# Personal Planning Agent - Behavioral Workflow Report

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

## User Profile
{user_profile_summary}

## Capacity Estimate
{capacity_summary}

## Planning Recommendation
{planning_summary}

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
    session_record = build_session_record(session_id, observation, score, patterns)
    history_memory.append(session_record.to_dict())
    save_history(HISTORY_FILE, history_memory)
    return report


def behavioral_agent(planned_behavior: str, actual_behavior: str) -> str:
    """Stable public callback used by the Gradio interface."""

    global history_memory
    history_memory = load_history(HISTORY_FILE)
    return run_workflow(planned_behavior, actual_behavior)


def _coerce_int(value: object, default: int = 0) -> int:
    try:
        if value is None or value == "":
            return default
        return int(float(value))
    except (TypeError, ValueError):
        return default


def parse_candidate_tasks(task_rows: object) -> List[CandidateTask]:
    headers = ["task_name", "estimated_minutes", "importance", "urgency"]
    if task_rows is None:
        return []

    if hasattr(task_rows, "to_dict"):
        task_rows = task_rows.to_dict(orient="records")

    if isinstance(task_rows, dict):
        if all(header in task_rows for header in headers):
            row_count = max(len(task_rows[header]) for header in headers)
            normalized_rows = []
            for index in range(row_count):
                normalized_rows.append(
                    {header: task_rows[header][index] for header in headers}
                )
            task_rows = normalized_rows
        else:
            return []

    parsed_tasks: List[CandidateTask] = []
    for row in task_rows:
        if isinstance(row, dict):
            task_name = str(row.get("task_name", "")).strip()
            estimated_minutes = _coerce_int(row.get("estimated_minutes"))
            importance = _coerce_int(row.get("importance"))
            urgency = _coerce_int(row.get("urgency"))
        else:
            row_values = list(row)
            if len(row_values) < 4:
                continue
            task_name = str(row_values[0]).strip()
            estimated_minutes = _coerce_int(row_values[1])
            importance = _coerce_int(row_values[2])
            urgency = _coerce_int(row_values[3])

        if not task_name or estimated_minutes <= 0:
            continue
        importance = max(1, min(5, importance or 1))
        urgency = max(1, min(5, urgency or 1))
        parsed_tasks.append(
            CandidateTask(
                task_name=task_name,
                estimated_minutes=estimated_minutes,
                importance=importance,
                urgency=urgency,
            )
        )

    return parsed_tasks


def planning_agent(available_minutes: int, candidate_tasks: object) -> str:
    report, _plan_state = planning_agent_with_state(available_minutes, candidate_tasks)
    return report


def planning_agent_with_state(available_minutes: int, candidate_tasks: object) -> tuple:
    available_minutes = max(0, _coerce_int(available_minutes))
    tasks = parse_candidate_tasks(candidate_tasks)
    if not tasks:
        return "# Personal Planning Report\n\nPlease enter at least one valid task.", {}

    history = load_history(HISTORY_FILE)
    profile = build_user_profile(history)
    planned_text = "; ".join(task.task_name for task in tasks)
    observation = ObservationLayer.collect(planned_text, "")
    capacity_estimate = CapacityEstimationEngine.estimate(profile, observation)
    task_level_plan = TaskPlanningEngine.generate_task_level_plan(
        available_minutes,
        tasks,
        capacity_estimate,
    )
    return generate_plan_report(task_level_plan), build_plan_state(task_level_plan)


def record_plan_outcome(
    plan_state: Dict[str, object],
    completed_tasks_text: str,
    actual_minutes: int,
    interruption_count: int,
    task_switch_count: int,
    fatigue_level: int,
    notes: str,
) -> str:
    completed_tasks = parse_completed_tasks(completed_tasks_text)
    outcome = build_plan_outcome_record(
        plan_state or {},
        completed_tasks,
        max(0, _coerce_int(actual_minutes)),
        max(0, _coerce_int(interruption_count)),
        max(0, _coerce_int(task_switch_count)),
        _coerce_int(fatigue_level, default=1),
        notes or "",
    )
    history = load_history(HISTORY_FILE)
    history.append(outcome.to_dict())
    save_history(HISTORY_FILE, history)
    return generate_plan_vs_outcome_report(outcome)


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
    with gr.Blocks(title="Personal Planning Agent") as interface:
        gr.Markdown("# Personal Planning Agent")
        gr.Markdown(
            "A rule-based prototype for reflective, multi-turn behavioural workflow analysis."
        )
        with gr.Tab("Behavioral Workflow"):
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

        with gr.Tab("Personal Planning Agent"):
            available_minutes = gr.Number(
                label="Available minutes",
                value=120,
                precision=0,
            )
            task_table = gr.Dataframe(
                headers=["task_name", "estimated_minutes", "importance", "urgency"],
                datatype=["str", "number", "number", "number"],
                type="array",
                row_count=4,
                column_count=4,
                value=[
                    ["STA2001 problem set", 60, 5, 5],
                    ["GRE vocabulary", 40, 4, 4],
                    ["Research reading", 30, 3, 2],
                    ["", None, None, None],
                ],
                label="Candidate tasks",
                interactive=True,
            )
            generate_plan_button = gr.Button("Generate Plan", variant="primary")
            plan_report = gr.Markdown(label="Personal Planning Report")
            plan_state = gr.State({})
            generate_plan_button.click(
                fn=planning_agent_with_state,
                inputs=[available_minutes, task_table],
                outputs=[plan_report, plan_state],
            )

            gr.Markdown("## Post-Session Feedback")
            completed_tasks = gr.Textbox(
                label="Actually completed tasks",
                lines=4,
                placeholder="e.g. STA2001 problem set, GRE vocabulary",
            )
            with gr.Row():
                actual_minutes = gr.Number(
                    label="Actual study minutes",
                    value=0,
                    precision=0,
                )
                interruption_count = gr.Number(
                    label="Interruptions",
                    value=0,
                    precision=0,
                )
                task_switch_count = gr.Number(
                    label="Task switches",
                    value=0,
                    precision=0,
                )
                fatigue_level = gr.Slider(
                    minimum=1,
                    maximum=5,
                    value=3,
                    step=1,
                    label="Fatigue level",
                )
            outcome_notes = gr.Textbox(
                label="Notes",
                lines=3,
                placeholder="Optional notes about what affected execution",
            )
            record_outcome_button = gr.Button("Record Outcome", variant="secondary")
            outcome_report = gr.Markdown(label="Plan vs Outcome Report")
            record_outcome_button.click(
                fn=record_plan_outcome,
                inputs=[
                    plan_state,
                    completed_tasks,
                    actual_minutes,
                    interruption_count,
                    task_switch_count,
                    fatigue_level,
                    outcome_notes,
                ],
                outputs=outcome_report,
            )
    return interface


demo = create_demo()


if __name__ == "__main__":
    demo.launch(css=load_css(), server_port=find_available_port())
