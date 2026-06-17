"""Shared data models for the Personal Planning Agent.

The models stay separate from the Gradio app so the planning workflow can be
tested, documented, and extended without turning app.py into the only source of
truth.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional


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
class OutcomeTrend:
    total_records: int
    total_generated_plans: int
    total_outcomes: int
    recent_completion_rate: Optional[float]
    average_planned_minutes: Optional[float]
    average_actual_minutes: Optional[float]
    average_interruptions: Optional[float]
    average_task_switches: Optional[float]
    average_fatigue: Optional[float]
    planning_adjustment: str
    risk_flags: List[str]


@dataclass(frozen=True)
class LLMReflection:
    enabled: bool
    provider: str
    model: str
    source: str
    reflection: str
    risk_explanation: str
    next_action: str
    follow_up_question: str
    error: Optional[str] = None


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

    def to_dict(self, plan_id: str) -> Dict[str, object]:
        return {
            "record_type": "task_level_plan",
            "plan_id": plan_id,
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "available_minutes": self.available_minutes,
            "work_minutes": self.work_minutes,
            "buffer_minutes": self.buffer_minutes,
            "planned_minutes": sum(
                task.allocated_minutes for task in self.selected_tasks
            ),
            "planned_task_count": len(self.selected_tasks),
            "selected_tasks": [
                {
                    "task_name": task.task_name,
                    "allocated_minutes": task.allocated_minutes,
                    "order": task.order,
                    "reason": task.reason,
                    "is_partial": task.is_partial,
                }
                for task in self.selected_tasks
            ],
            "deferred_tasks": [
                {
                    "task_name": task.task_name,
                    "estimated_minutes": task.estimated_minutes,
                    "defer_type": task.defer_type,
                    "reason": task.reason,
                }
                for task in self.deferred_tasks
            ],
            "planning_risks": self.planning_risks,
            "plan_confidence": self.plan_confidence,
            "patterns": ["TaskLevelPlan"],
        }


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
