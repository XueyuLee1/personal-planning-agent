import importlib.util
import sys
import os
import tempfile
import unittest
from pathlib import Path


APP_PATH = Path(__file__).resolve().parents[1] / "app.py"
SPEC = importlib.util.spec_from_file_location("behavior_app", APP_PATH)
app = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = app
SPEC.loader.exec_module(app)

MEMORY_PATH = Path(__file__).resolve().parents[1] / "temporal_memory.py"
MEMORY_SPEC = importlib.util.spec_from_file_location("temporal_memory", MEMORY_PATH)
temporal_memory = importlib.util.module_from_spec(MEMORY_SPEC)
sys.modules[MEMORY_SPEC.name] = temporal_memory
MEMORY_SPEC.loader.exec_module(temporal_memory)


class DiagnosticTests(unittest.TestCase):
    def test_stable_execution_gets_positive_execution_delta(self):
        observation = app.ObservationLayer.collect(
            "Read paper; draft notes",
            "Read paper; draft notes",
        )

        result = app.ExecutionGapAnalyzer().analyze(observation)

        self.assertEqual(result.score_delta, 8)

    def test_distraction_detector_penalizes_distraction_terms(self):
        observation = app.ObservationLayer.collect(
            "Write experiment plan; review notes",
            "Opened notes, then watched YouTube and scrolled Reddit",
        )

        result = app.DistractionDetector().analyze(observation)

        self.assertLessEqual(result.score_delta, -20)
        self.assertEqual(result.pattern_name, "DistractionDetector")

    def test_execution_gap_detects_under_execution(self):
        observation = app.ObservationLayer.collect(
            "Read paper; draft notes; write outline; prepare questions",
            "Read paper",
        )

        result = app.ExecutionGapAnalyzer().analyze(observation)

        self.assertEqual(result.score_delta, -28)

    def test_planning_mismatch_detects_low_overlap(self):
        observation = app.ObservationLayer.collect(
            "Study prompt optimisation and meta learning",
            "Cleaned desk and answered messages",
        )

        result = app.PlanningMismatchAnalyzer().analyze(observation)

        self.assertEqual(result.score_delta, -22)

    def test_task_switch_analyzer_penalizes_over_switching(self):
        observation = app.ObservationLayer.collect(
            "Read paper, draft notes",
            "Read abstract, check email, answer message, open notes, skim paper, update todo",
        )

        result = app.TaskSwitchAnalyzer().analyze(observation)

        self.assertLess(result.score_delta, 0)
        self.assertIn("TaskSwitchAnalyzer", result.pattern_name)

    def test_persistent_history_round_trip_keeps_two_sessions(self):
        with tempfile.NamedTemporaryFile() as history_file:
            records = [
                {"score": 80, "patterns": ["StableExecution"]},
                {"score": 55, "patterns": ["DistractionDetector"]},
            ]

            temporal_memory.save_history(history_file.name, records)
            loaded = temporal_memory.load_history(history_file.name)

        self.assertEqual(len(loaded), 2)
        self.assertEqual(loaded[1]["patterns"], ["DistractionDetector"])

    def test_load_history_keeps_legacy_records(self):
        with tempfile.NamedTemporaryFile() as history_file:
            legacy_records = [{"score": 70, "patterns": ["ExecutionGapAnalyzer"]}]

            temporal_memory.save_history(history_file.name, legacy_records)
            loaded = temporal_memory.load_history(history_file.name)

        self.assertEqual(loaded, legacy_records)

    def test_load_history_keeps_generated_plan_records(self):
        with tempfile.NamedTemporaryFile() as history_file:
            records = [
                {
                    "record_type": "task_level_plan",
                    "plan_id": "P001",
                    "selected_tasks": [],
                    "patterns": ["TaskLevelPlan"],
                }
            ]

            temporal_memory.save_history(history_file.name, records)
            loaded = temporal_memory.load_history(history_file.name)

        self.assertEqual(loaded, records)

    def test_build_session_record_contains_structured_fields(self):
        observation = app.ObservationLayer.collect(
            "Read paper; draft notes",
            "Read paper",
        )

        record = app.build_session_record(
            "S001-test",
            observation,
            72,
            ["ExecutionGapAnalyzer"],
        ).to_dict()

        self.assertEqual(record["session_id"], "S001-test")
        self.assertEqual(record["planned_task_count"], 2)
        self.assertEqual(record["actual_task_count"], 1)
        self.assertEqual(record["planned_text"], "Read paper; draft notes")
        self.assertEqual(record["actual_text"], "Read paper")
        self.assertEqual(record["score"], 72)

    def test_user_profile_handles_empty_history(self):
        profile = app.build_user_profile([])

        self.assertEqual(profile.total_sessions, 0)
        self.assertIsNone(profile.average_score)
        self.assertEqual(profile.common_patterns, [])
        self.assertEqual(profile.recent_score_trend, "not enough data")

    def test_user_profile_summarizes_mixed_history(self):
        history = [
            {"score": 80, "patterns": ["StableExecution"]},
            {
                "score": 60,
                "patterns": ["ExecutionGapAnalyzer"],
                "planned_task_count": 4,
                "actual_task_count": 2,
            },
            {
                "score": 70,
                "patterns": ["ExecutionGapAnalyzer"],
                "planned_task_count": 3,
                "actual_task_count": 3,
            },
        ]

        profile = app.build_user_profile(history)

        self.assertEqual(profile.total_sessions, 3)
        self.assertAlmostEqual(profile.average_score, 70.0)
        self.assertAlmostEqual(profile.average_planned_tasks, 3.5)
        self.assertAlmostEqual(profile.average_actual_tasks, 2.5)
        self.assertAlmostEqual(profile.average_completion_rate, 0.75)
        self.assertAlmostEqual(profile.overplanning_frequency, 0.5)
        self.assertEqual(profile.common_patterns[0], "ExecutionGapAnalyzer")

    def test_report_includes_user_profile_section(self):
        observation = app.ObservationLayer.collect(
            "Read paper; draft notes",
            "Read paper",
        )
        diagnostics = [app.ExecutionGapAnalyzer().analyze(observation)]

        report = app.build_report(
            "S001-test",
            72,
            ["ExecutionGapAnalyzer"],
            diagnostics,
            observation,
            [{"score": 80, "patterns": ["StableExecution"]}],
        )

        self.assertIn("## User Profile", report)
        self.assertIn("Total recorded sessions: 1", report)

    def test_capacity_estimate_handles_empty_profile(self):
        profile = app.build_user_profile([])
        observation = app.ObservationLayer.collect(
            "Read paper; draft notes",
            "Read paper",
        )

        estimate = app.CapacityEstimationEngine.estimate(profile, observation)

        self.assertIsNone(estimate.estimated_task_capacity)
        self.assertEqual(estimate.confidence, "low")
        self.assertEqual(estimate.reliable_task_range, "not enough data")

    def test_capacity_estimate_uses_historical_actual_tasks(self):
        profile = app.build_user_profile(
            [
                {
                    "score": 70,
                    "patterns": ["ExecutionGapAnalyzer"],
                    "planned_task_count": 4,
                    "actual_task_count": 2,
                },
                {
                    "score": 75,
                    "patterns": ["StableExecution"],
                    "planned_task_count": 3,
                    "actual_task_count": 3,
                },
                {
                    "score": 80,
                    "patterns": ["StableExecution"],
                    "planned_task_count": 3,
                    "actual_task_count": 2,
                },
            ]
        )
        observation = app.ObservationLayer.collect(
            "Read paper; draft notes",
            "Read paper",
        )

        estimate = app.CapacityEstimationEngine.estimate(profile, observation)

        self.assertEqual(estimate.estimated_task_capacity, 2)
        self.assertIn("tasks", estimate.reliable_task_range)

    def test_capacity_estimate_flags_plan_above_reliable_range(self):
        profile = app.build_user_profile(
            [
                {
                    "score": 60,
                    "patterns": ["ExecutionGapAnalyzer"],
                    "planned_task_count": 5,
                    "actual_task_count": 2,
                },
                {
                    "score": 62,
                    "patterns": ["ExecutionGapAnalyzer"],
                    "planned_task_count": 4,
                    "actual_task_count": 2,
                },
                {
                    "score": 64,
                    "patterns": ["ExecutionGapAnalyzer"],
                    "planned_task_count": 5,
                    "actual_task_count": 1,
                },
            ]
        )
        observation = app.ObservationLayer.collect(
            "Read paper; draft notes; write outline; prepare questions",
            "Read paper",
        )

        estimate = app.CapacityEstimationEngine.estimate(profile, observation)

        self.assertTrue(
            any("above the estimated reliable range" in note for note in estimate.risk_notes)
        )

    def test_report_includes_capacity_estimate_section(self):
        observation = app.ObservationLayer.collect(
            "Read paper; draft notes",
            "Read paper",
        )
        diagnostics = [app.ExecutionGapAnalyzer().analyze(observation)]

        report = app.build_report(
            "S001-test",
            72,
            ["ExecutionGapAnalyzer"],
            diagnostics,
            observation,
            [
                {
                    "score": 70,
                    "patterns": ["ExecutionGapAnalyzer"],
                    "planned_task_count": 3,
                    "actual_task_count": 2,
                }
            ],
        )

        self.assertIn("## Capacity Estimate", report)
        self.assertIn("Estimated task capacity", report)

    def test_planning_recommendation_is_conservative_without_history(self):
        profile = app.build_user_profile([])
        observation = app.ObservationLayer.collect(
            "Read paper; draft notes; write outline",
            "",
        )
        capacity = app.CapacityEstimationEngine.estimate(profile, observation)

        recommendation = app.PlanningRecommendationEngine.recommend(
            observation,
            profile,
            capacity,
        )

        self.assertEqual(recommendation.recommended_task_count, 2)
        self.assertEqual(len(recommendation.recommended_tasks), 2)
        self.assertLessEqual(recommendation.success_probability, 65)

    def test_planning_recommendation_caps_tasks_by_capacity(self):
        profile = app.build_user_profile(
            [
                {
                    "score": 70,
                    "patterns": ["ExecutionGapAnalyzer"],
                    "planned_task_count": 4,
                    "actual_task_count": 2,
                },
                {
                    "score": 72,
                    "patterns": ["StableExecution"],
                    "planned_task_count": 3,
                    "actual_task_count": 2,
                },
                {
                    "score": 74,
                    "patterns": ["StableExecution"],
                    "planned_task_count": 3,
                    "actual_task_count": 2,
                },
            ]
        )
        observation = app.ObservationLayer.collect(
            "Read paper; draft notes; write outline; prepare questions",
            "Read paper",
        )
        capacity = app.CapacityEstimationEngine.estimate(profile, observation)

        recommendation = app.PlanningRecommendationEngine.recommend(
            observation,
            profile,
            capacity,
        )

        self.assertLessEqual(
            recommendation.recommended_task_count,
            capacity.estimated_task_capacity,
        )
        self.assertEqual(len(recommendation.recommended_tasks), recommendation.recommended_task_count)
        self.assertTrue(10 <= recommendation.success_probability <= 95)

    def test_planning_recommendation_reports_risk_when_plan_is_reduced(self):
        profile = app.build_user_profile(
            [
                {
                    "score": 60,
                    "patterns": ["ExecutionGapAnalyzer"],
                    "planned_task_count": 5,
                    "actual_task_count": 1,
                },
                {
                    "score": 62,
                    "patterns": ["ExecutionGapAnalyzer"],
                    "planned_task_count": 4,
                    "actual_task_count": 1,
                },
                {
                    "score": 64,
                    "patterns": ["ExecutionGapAnalyzer"],
                    "planned_task_count": 4,
                    "actual_task_count": 1,
                },
            ]
        )
        observation = app.ObservationLayer.collect(
            "Read paper; draft notes; write outline; prepare questions",
            "Read paper",
        )
        capacity = app.CapacityEstimationEngine.estimate(profile, observation)

        recommendation = app.PlanningRecommendationEngine.recommend(
            observation,
            profile,
            capacity,
        )

        self.assertTrue(
            any("recommendation limits it" in warning for warning in recommendation.risk_warnings)
        )

    def test_report_includes_planning_recommendation_section(self):
        observation = app.ObservationLayer.collect(
            "Read paper; draft notes",
            "Read paper",
        )
        diagnostics = [app.ExecutionGapAnalyzer().analyze(observation)]

        report = app.build_report(
            "S001-test",
            72,
            ["ExecutionGapAnalyzer"],
            diagnostics,
            observation,
            [
                {
                    "score": 70,
                    "patterns": ["ExecutionGapAnalyzer"],
                    "planned_task_count": 3,
                    "actual_task_count": 2,
                }
            ],
        )

        self.assertIn("## Planning Recommendation", report)
        self.assertIn("Estimated success probability", report)

    def test_task_planning_scores_importance_and_urgency(self):
        task = app.CandidateTask(
            task_name="Write research notes",
            estimated_minutes=45,
            importance=5,
            urgency=3,
        )

        scored = app.TaskPlanningEngine.score_task(task, available_minutes=60)

        self.assertAlmostEqual(scored.priority_score, 4.2)
        self.assertEqual(scored.feasibility_score, 1.0)
        self.assertGreater(scored.final_score, 0)

    def test_task_selection_selects_tasks_within_available_time(self):
        tasks = [
            app.CandidateTask("Long project", 90, 5, 5),
            app.CandidateTask("Reading", 30, 4, 4),
            app.CandidateTask("Review notes", 20, 3, 3),
        ]
        capacity = app.CapacityEstimate(
            estimated_task_capacity=2,
            reliable_task_range="1-2 tasks",
            confidence="medium",
            basis="test",
            risk_notes=[],
        )

        result = app.TaskPlanningEngine.select_tasks(
            available_minutes=60,
            candidate_tasks=tasks,
            capacity_estimate=capacity,
        )

        selected_minutes = sum(task.estimated_minutes for task in result.selected_tasks)
        self.assertLessEqual(selected_minutes, 60)
        self.assertLessEqual(len(result.selected_tasks), 2)
        self.assertIn("Long project", [task.task_name for task in result.deferred_tasks])

    def test_task_selection_respects_capacity_limit(self):
        tasks = [
            app.CandidateTask("Task A", 20, 5, 5),
            app.CandidateTask("Task B", 20, 4, 4),
            app.CandidateTask("Task C", 20, 3, 3),
        ]
        capacity = app.CapacityEstimate(
            estimated_task_capacity=1,
            reliable_task_range="1-1 tasks",
            confidence="high",
            basis="test",
            risk_notes=[],
        )

        result = app.TaskPlanningEngine.select_tasks(
            available_minutes=90,
            candidate_tasks=tasks,
            capacity_estimate=capacity,
        )

        self.assertEqual(len(result.selected_tasks), 1)
        self.assertEqual(len(result.deferred_tasks), 2)

    def test_task_selection_defaults_to_two_tasks_without_capacity_history(self):
        tasks = [
            app.CandidateTask("Task A", 10, 5, 5),
            app.CandidateTask("Task B", 10, 4, 4),
            app.CandidateTask("Task C", 10, 3, 3),
        ]
        capacity = app.CapacityEstimate(
            estimated_task_capacity=None,
            reliable_task_range="not enough data",
            confidence="low",
            basis="test",
            risk_notes=[],
        )

        result = app.TaskPlanningEngine.select_tasks(
            available_minutes=90,
            candidate_tasks=tasks,
            capacity_estimate=capacity,
        )

        self.assertEqual(len(result.selected_tasks), 2)
        self.assertEqual(len(result.deferred_tasks), 1)

    def test_task_planning_calculates_buffer_and_work_time(self):
        buffer_info = app.TaskPlanningEngine.calculate_buffer(120)

        self.assertEqual(buffer_info["buffer_minutes"], 18)
        self.assertEqual(buffer_info["work_minutes"], 102)

    def test_task_planning_orders_by_urgency_importance_and_shorter_time(self):
        tasks = [
            app.CandidateTask("Reading", 30, 3, 2),
            app.CandidateTask("GRE", 40, 3, 5),
            app.CandidateTask("STA2001", 60, 5, 5),
        ]

        ordered = app.TaskPlanningEngine.order_tasks(tasks)

        self.assertEqual([task.task_name for task in ordered], ["STA2001", "GRE", "Reading"])

    def test_task_planning_allocates_time_to_plan_items(self):
        tasks = [
            app.CandidateTask("STA2001", 60, 5, 5),
            app.CandidateTask("GRE", 40, 4, 4),
        ]

        items = app.TaskPlanningEngine.allocate_time(tasks, 100)

        self.assertEqual(items[0].task_name, "STA2001")
        self.assertEqual(items[0].allocated_minutes, 60)
        self.assertFalse(items[0].is_partial)
        self.assertEqual(items[1].allocated_minutes, 40)

    def test_task_planning_supports_partial_allocation(self):
        tasks = [
            app.CandidateTask("Long paper", 90, 5, 5),
        ]

        items = app.TaskPlanningEngine.allocate_time(tasks, 50)

        self.assertEqual(items[0].allocated_minutes, 50)
        self.assertTrue(items[0].is_partial)
        self.assertIn("Partial progress", items[0].reason)

    def test_deferred_task_reasoning_classifies_time_limit(self):
        deferred = [app.CandidateTask("Long project", 120, 5, 5)]
        capacity = app.CapacityEstimate(
            estimated_task_capacity=3,
            reliable_task_range="1-3 tasks",
            confidence="high",
            basis="test",
            risk_notes=[],
        )

        explained = app.TaskPlanningEngine.explain_deferred_tasks(
            deferred,
            selected_tasks=[],
            work_minutes=60,
            capacity_estimate=capacity,
        )

        self.assertEqual(explained[0].defer_type, "time_limit")
        self.assertIn("exceeds today's work budget", explained[0].reason)

    def test_deferred_task_reasoning_classifies_capacity_limit(self):
        selected = [app.CandidateTask("Task A", 20, 5, 5)]
        deferred = [app.CandidateTask("Task B", 20, 4, 4)]
        capacity = app.CapacityEstimate(
            estimated_task_capacity=1,
            reliable_task_range="1-1 tasks",
            confidence="high",
            basis="test",
            risk_notes=[],
        )

        explained = app.TaskPlanningEngine.explain_deferred_tasks(
            deferred,
            selected_tasks=selected,
            work_minutes=90,
            capacity_estimate=capacity,
        )

        self.assertEqual(explained[0].defer_type, "capacity_limit")
        self.assertIn("historical capacity", explained[0].reason)

    def test_deferred_task_reasoning_classifies_low_priority(self):
        deferred = [app.CandidateTask("Optional reading", 20, 1, 1)]
        capacity = app.CapacityEstimate(
            estimated_task_capacity=3,
            reliable_task_range="1-3 tasks",
            confidence="high",
            basis="test",
            risk_notes=[],
        )

        explained = app.TaskPlanningEngine.explain_deferred_tasks(
            deferred,
            selected_tasks=[],
            work_minutes=90,
            capacity_estimate=capacity,
        )

        self.assertEqual(explained[0].defer_type, "low_priority")

    def test_task_level_plan_generation(self):
        tasks = [
            app.CandidateTask("STA2001", 60, 5, 5),
            app.CandidateTask("GRE", 40, 4, 4),
            app.CandidateTask("Reading", 30, 3, 2),
        ]
        capacity = app.CapacityEstimate(
            estimated_task_capacity=2,
            reliable_task_range="1-2 tasks",
            confidence="medium",
            basis="test",
            risk_notes=[],
        )

        plan = app.TaskPlanningEngine.generate_task_level_plan(
            available_minutes=120,
            candidate_tasks=tasks,
            capacity_estimate=capacity,
        )

        self.assertEqual(plan.available_minutes, 120)
        self.assertEqual(plan.buffer_minutes, 18)
        self.assertEqual(plan.work_minutes, 102)
        self.assertLessEqual(len(plan.selected_tasks), 2)
        self.assertTrue(all(isinstance(item, app.TaskPlanItem) for item in plan.selected_tasks))
        self.assertTrue(all(isinstance(task, app.DeferredTask) for task in plan.deferred_tasks))
        self.assertIn(plan.plan_confidence, {"high", "medium", "low"})

    def test_task_level_plan_confidence_low_without_capacity_history(self):
        tasks = [app.CandidateTask("Task A", 20, 5, 5)]
        capacity = app.CapacityEstimate(
            estimated_task_capacity=None,
            reliable_task_range="not enough data",
            confidence="low",
            basis="test",
            risk_notes=[],
        )

        plan = app.TaskPlanningEngine.generate_task_level_plan(
            available_minutes=60,
            candidate_tasks=tasks,
            capacity_estimate=capacity,
        )

        self.assertEqual(plan.plan_confidence, "low")

    def test_generate_plan_report_contains_required_sections(self):
        plan = app.TaskLevelPlan(
            available_minutes=120,
            work_minutes=102,
            buffer_minutes=18,
            selected_tasks=[
                app.TaskPlanItem("STA2001", 60, 1, "Selected for high priority.", False)
            ],
            deferred_tasks=[
                app.DeferredTask("Reading", 30, "low_priority", "Deferred because it was lower priority.")
            ],
            planning_risks=["Some candidate tasks were deferred."],
            plan_confidence="medium",
        )

        report = app.generate_plan_report(plan)

        self.assertIn("Personal Planning Report", report)
        self.assertIn("Available Time", report)
        self.assertIn("Protected Buffer", report)
        self.assertIn("Selected Tasks", report)
        self.assertIn("Deferred Tasks", report)
        self.assertIn("Plan Confidence", report)

    def test_parse_candidate_tasks_from_table_rows(self):
        rows = [
            ["STA2001", 60, 5, 5],
            ["GRE", "40", "4", "3"],
            ["", 30, 3, 2],
            ["Invalid", 0, 3, 3],
        ]

        tasks = app.parse_candidate_tasks(rows)

        self.assertEqual(len(tasks), 2)
        self.assertEqual(tasks[0].task_name, "STA2001")
        self.assertEqual(tasks[1].estimated_minutes, 40)
        self.assertEqual(tasks[1].importance, 4)

    def test_parse_candidate_tasks_clamps_importance_and_urgency(self):
        rows = [["Task A", 20, 9, 0]]

        tasks = app.parse_candidate_tasks(rows)

        self.assertEqual(tasks[0].importance, 5)
        self.assertEqual(tasks[0].urgency, 1)

    def test_planning_agent_generates_personal_planning_report(self):
        original_history_file = app.HISTORY_FILE

        with tempfile.NamedTemporaryFile() as history_file:
            app.HISTORY_FILE = history_file.name
            report = app.planning_agent(
                120,
                [
                    ["STA2001", 60, 5, 5],
                    ["GRE", 40, 4, 4],
                    ["Reading", 30, 3, 2],
                ],
            )
            app.HISTORY_FILE = original_history_file

        self.assertIn("Personal Planning Report", report)
        self.assertIn("Available Time", report)
        self.assertIn("Selected Tasks", report)
        self.assertIn("Plan Confidence", report)

    def test_planning_agent_handles_empty_task_rows(self):
        report = app.planning_agent(120, [["", None, None, None]])

        self.assertIn("Please enter at least one valid task", report)

    def test_plan_outcome_record_creation(self):
        outcome = app.build_plan_outcome_record(
            {
                "plan_id": "P001-test",
                "planned_tasks": ["STA2001", "GRE"],
                "planned_minutes": 100,
            },
            completed_tasks=["STA2001"],
            actual_minutes=80,
            interruption_count=2,
            task_switch_count=3,
            fatigue_level=4,
            notes="Tired near the end",
        )

        self.assertEqual(outcome.plan_id, "P001-test")
        self.assertEqual(outcome.planned_minutes, 100)
        self.assertEqual(outcome.actual_minutes, 80)
        self.assertAlmostEqual(outcome.completion_rate, 0.5)
        self.assertEqual(outcome.fatigue_level, 4)

    def test_completion_rate_calculation(self):
        rate = app.calculate_completion_rate(
            ["STA2001", "GRE", "Reading"],
            ["sta2001", "Reading"],
        )

        self.assertAlmostEqual(rate, 2 / 3)

    def test_plan_vs_outcome_report_contains_core_fields(self):
        outcome = app.PlanOutcomeRecord(
            plan_id="P001-test",
            created_at="2026-06-15T10:00:00",
            planned_tasks=["STA2001", "GRE"],
            planned_minutes=100,
            completed_tasks=["STA2001"],
            actual_minutes=80,
            interruption_count=2,
            task_switch_count=3,
            fatigue_level=4,
            completion_rate=0.5,
            notes="",
        )

        report = app.generate_plan_vs_outcome_report(outcome)

        self.assertIn("Plan vs Outcome Report", report)
        self.assertIn("Planned Tasks", report)
        self.assertIn("Completed Tasks", report)
        self.assertIn("Completion Rate", report)
        self.assertIn("Interruptions", report)

    def test_record_plan_outcome_saves_to_history(self):
        original_history_file = app.HISTORY_FILE

        with tempfile.NamedTemporaryFile() as history_file:
            app.HISTORY_FILE = history_file.name
            report = app.record_plan_outcome(
                {
                    "plan_id": "P001-test",
                    "planned_tasks": ["STA2001", "GRE"],
                    "planned_minutes": 100,
                },
                "STA2001",
                80,
                1,
                2,
                3,
                "ok",
            )
            loaded = temporal_memory.load_history(history_file.name)
            app.HISTORY_FILE = original_history_file

        self.assertIn("Plan vs Outcome Report", report)
        self.assertEqual(len(loaded), 1)
        self.assertEqual(loaded[0]["record_type"], "plan_outcome")
        self.assertIn("score", loaded[0])
        self.assertIn("patterns", loaded[0])

    def test_empty_feedback_is_handled(self):
        outcome = app.build_plan_outcome_record(
            {},
            completed_tasks=[],
            actual_minutes=0,
            interruption_count=0,
            task_switch_count=0,
            fatigue_level=0,
            notes="",
        )

        self.assertEqual(outcome.planned_tasks, [])
        self.assertEqual(outcome.completion_rate, 0.0)
        self.assertEqual(outcome.fatigue_level, 1)

    def test_planning_agent_with_state_returns_plan_state(self):
        original_history_file = app.HISTORY_FILE

        with tempfile.NamedTemporaryFile() as history_file:
            app.HISTORY_FILE = history_file.name
            report, state = app.planning_agent_with_state(
                120,
                [
                    ["STA2001", 60, 5, 5],
                    ["GRE", 40, 4, 4],
                ],
            )
            app.HISTORY_FILE = original_history_file

        self.assertIn("Personal Planning Report", report)
        self.assertIn("planned_tasks", state)
        self.assertIn("planned_minutes", state)
        self.assertIn("plan_confidence", state)

    def test_generated_plan_is_saved_to_history(self):
        original_history_file = app.HISTORY_FILE

        with tempfile.NamedTemporaryFile() as history_file:
            app.HISTORY_FILE = history_file.name
            _report, state = app.planning_agent_with_state(
                120,
                [
                    ["STA2001", 60, 5, 5],
                    ["GRE", 40, 4, 4],
                ],
            )
            loaded = temporal_memory.load_history(history_file.name)
            app.HISTORY_FILE = original_history_file

        self.assertEqual(len(loaded), 1)
        self.assertEqual(loaded[0]["record_type"], "task_level_plan")
        self.assertEqual(loaded[0]["plan_id"], state["plan_id"])
        self.assertIn("selected_tasks", loaded[0])
        self.assertIn("deferred_tasks", loaded[0])

    def test_user_profile_uses_plan_outcome_records(self):
        history = [
            {
                "record_type": "plan_outcome",
                "planned_tasks": ["STA2001", "GRE"],
                "completed_tasks": ["STA2001"],
                "completion_rate": 0.5,
                "score": 50,
                "patterns": ["PlanOutcomeRecord"],
            }
        ]

        profile = app.build_user_profile(history)

        self.assertEqual(profile.total_sessions, 1)
        self.assertAlmostEqual(profile.average_planned_tasks, 2.0)
        self.assertAlmostEqual(profile.average_actual_tasks, 1.0)
        self.assertAlmostEqual(profile.average_completion_rate, 0.5)

    def test_history_trend_summary_reports_recent_outcomes(self):
        history = [
            {
                "record_type": "task_level_plan",
                "plan_id": "P001",
                "patterns": ["TaskLevelPlan"],
            },
            {
                "record_type": "plan_outcome",
                "planned_minutes": 100,
                "actual_minutes": 80,
                "completion_rate": 0.5,
                "interruption_count": 2,
                "task_switch_count": 3,
                "fatigue_level": 4,
                "patterns": ["PlanOutcomeRecord"],
            },
        ]

        summary = app.generate_history_trend_summary(history)

        self.assertIn("Generated plans: 1", summary)
        self.assertIn("Recorded outcomes: 1", summary)
        self.assertIn("Recent outcome completion rate: 50%", summary)

    def test_outcome_trend_reduces_work_budget_after_repeated_low_completion(self):
        trend = app.build_outcome_trend(
            [
                {
                    "record_type": "plan_outcome",
                    "planned_minutes": 100,
                    "actual_minutes": 40,
                    "completion_rate": 0.4,
                    "interruption_count": 4,
                    "task_switch_count": 6,
                    "fatigue_level": 4,
                    "patterns": ["PlanOutcomeRecord"],
                }
                for _index in range(3)
            ]
        )

        adjustment = app.TaskPlanningEngine.adjust_work_minutes_by_trend(100, trend)

        self.assertLess(adjustment["work_minutes"], 100)
        self.assertIn("reduced", adjustment["adjustment_note"])

    def test_task_level_plan_uses_outcome_trend_adjustment(self):
        trend = app.build_outcome_trend(
            [
                {
                    "record_type": "plan_outcome",
                    "planned_minutes": 100,
                    "actual_minutes": 40,
                    "completion_rate": 0.4,
                    "interruption_count": 4,
                    "task_switch_count": 6,
                    "fatigue_level": 4,
                    "patterns": ["PlanOutcomeRecord"],
                }
                for _index in range(3)
            ]
        )
        capacity = app.CapacityEstimate(
            estimated_task_capacity=3,
            reliable_task_range="2-3 tasks",
            confidence="high",
            basis="test",
            risk_notes=[],
        )

        plan = app.TaskPlanningEngine.generate_task_level_plan(
            120,
            [
                app.CandidateTask("Task A", 40, 5, 5),
                app.CandidateTask("Task B", 40, 4, 4),
                app.CandidateTask("Task C", 40, 3, 3),
            ],
            capacity,
            trend,
        )

        self.assertLess(plan.work_minutes, 102)
        self.assertTrue(any("reduced" in risk for risk in plan.planning_risks))

    def test_history_dashboard_contains_profile_and_trend(self):
        original_history_file = app.HISTORY_FILE

        with tempfile.NamedTemporaryFile() as history_file:
            app.HISTORY_FILE = history_file.name
            temporal_memory.save_history(
                history_file.name,
                [
                    {
                        "record_type": "plan_outcome",
                        "planned_tasks": ["A", "B"],
                        "completed_tasks": ["A"],
                        "planned_minutes": 100,
                        "actual_minutes": 70,
                        "completion_rate": 0.5,
                        "interruption_count": 1,
                        "task_switch_count": 2,
                        "fatigue_level": 3,
                        "score": 50,
                        "patterns": ["PlanOutcomeRecord"],
                    }
                ],
            )
            dashboard = app.generate_history_dashboard()
            app.HISTORY_FILE = original_history_file

        self.assertIn("Planning History Dashboard", dashboard)
        self.assertIn("Outcome Trend", dashboard)
        self.assertIn("User Profile", dashboard)

    def test_llm_reflection_agent_falls_back_without_deepseek_api_key(self):
        original_provider = os.environ.get("LLM_PROVIDER")
        original_api_key = os.environ.pop("DEEPSEEK_API_KEY", None)
        os.environ["LLM_PROVIDER"] = "deepseek"
        plan = app.TaskLevelPlan(
            available_minutes=120,
            work_minutes=102,
            buffer_minutes=18,
            selected_tasks=[
                app.TaskPlanItem("STA2001", 60, 1, "Selected", False)
            ],
            deferred_tasks=[],
            planning_risks=["No major planning risks detected."],
            plan_confidence="low",
        )
        profile = app.UserProfile(
            total_sessions=0,
            average_score=None,
            average_planned_tasks=None,
            average_actual_tasks=None,
            average_completion_rate=None,
            overplanning_frequency=None,
            common_patterns=[],
            recent_score_trend="not enough data",
        )
        capacity = app.CapacityEstimate(
            estimated_task_capacity=None,
            reliable_task_range="not enough data",
            confidence="low",
            basis="test",
            risk_notes=["not enough data"],
        )
        trend = app.build_outcome_trend([])

        reflection = app.LLMReflectionAgent.generate(profile, capacity, trend, plan)

        if original_api_key is not None:
            os.environ["DEEPSEEK_API_KEY"] = original_api_key
        if original_provider is not None:
            os.environ["LLM_PROVIDER"] = original_provider
        else:
            os.environ.pop("LLM_PROVIDER", None)
        self.assertEqual(reflection.source, "rule_based_fallback")
        self.assertEqual(reflection.provider, "deepseek")
        self.assertIn("DEEPSEEK_API_KEY", reflection.error)
        self.assertTrue(reflection.reflection)

    def test_deepseek_chat_completion_text_extraction(self):
        payload = {
            "choices": [
                {
                    "message": {
                        "content": '{"reflection":"a","risk_explanation":"b","next_action":"c","follow_up_question":"d"}'
                    }
                }
            ]
        }

        text = app.LLMReflectionAgent._extract_chat_completion_text(payload)

        self.assertIn('"reflection"', text)

    def test_invalid_llm_provider_falls_back_to_deepseek(self):
        original_provider = os.environ.get("LLM_PROVIDER")
        original_api_key = os.environ.pop("DEEPSEEK_API_KEY", None)
        os.environ["LLM_PROVIDER"] = "unknown-provider"
        plan = app.TaskLevelPlan(
            available_minutes=60,
            work_minutes=51,
            buffer_minutes=9,
            selected_tasks=[app.TaskPlanItem("Task A", 30, 1, "Selected", False)],
            deferred_tasks=[],
            planning_risks=["No major planning risks detected."],
            plan_confidence="low",
        )
        profile = app.UserProfile(
            total_sessions=0,
            average_score=None,
            average_planned_tasks=None,
            average_actual_tasks=None,
            average_completion_rate=None,
            overplanning_frequency=None,
            common_patterns=[],
            recent_score_trend="not enough data",
        )
        capacity = app.CapacityEstimate(
            estimated_task_capacity=None,
            reliable_task_range="not enough data",
            confidence="low",
            basis="test",
            risk_notes=["not enough data"],
        )

        reflection = app.LLMReflectionAgent.generate(
            profile, capacity, app.build_outcome_trend([]), plan
        )

        if original_provider is not None:
            os.environ["LLM_PROVIDER"] = original_provider
        else:
            os.environ.pop("LLM_PROVIDER", None)
        if original_api_key is not None:
            os.environ["DEEPSEEK_API_KEY"] = original_api_key

        self.assertEqual(reflection.provider, "deepseek")
        self.assertEqual(reflection.source, "rule_based_fallback")

    def test_plan_report_contains_llm_reflection_section(self):
        reflection = app.LLMReflection(
            enabled=True,
            provider="deepseek",
            model="test-model",
            source="rule_based_fallback",
            reflection="Plan keeps the workload small.",
            risk_explanation="Not enough historical data.",
            next_action="Start with task one.",
            follow_up_question="What blocked execution?",
            error="DEEPSEEK_API_KEY is not set.",
        )
        plan = app.TaskLevelPlan(
            available_minutes=120,
            work_minutes=102,
            buffer_minutes=18,
            selected_tasks=[
                app.TaskPlanItem("STA2001", 60, 1, "Selected", False)
            ],
            deferred_tasks=[],
            planning_risks=["No major planning risks detected."],
            plan_confidence="low",
        )

        report = app.generate_plan_report_with_history(plan, "No history.", reflection)

        self.assertIn("LLM Reflection Agent", report)
        self.assertIn("rule_based_fallback", report)
        self.assertIn("Plan keeps the workload small.", report)

    def test_planning_agent_can_disable_llm_reflection(self):
        original_history_file = app.HISTORY_FILE

        with tempfile.NamedTemporaryFile() as history_file:
            app.HISTORY_FILE = history_file.name
            report, _state = app.planning_agent_with_state(
                120,
                [["STA2001", 60, 5, 5]],
                enable_llm_reflection=False,
            )
            app.HISTORY_FILE = original_history_file

        self.assertIn("LLM Reflection Agent", report)
        self.assertIn("disabled", report)

    def test_progress_bar_for_score_50_has_ten_filled_segments(self):
        progress_bar = app.build_progress_bar(50)

        self.assertEqual(progress_bar.count("█"), 10)
        self.assertEqual(progress_bar.count("─"), 10)
        self.assertEqual(len(progress_bar), 20)

    def test_history_records_patterns_after_agent_run(self):
        app.history_memory.clear()
        original_history_file = app.HISTORY_FILE

        with tempfile.NamedTemporaryFile() as history_file:
            app.HISTORY_FILE = history_file.name

            report = app.behavioral_agent(
                "Read paper; draft notes; write outline",
                "Watched YouTube and answered messages",
            )

            app.HISTORY_FILE = original_history_file

        self.assertIn("Temporal Pattern Memory", report)
        self.assertTrue(app.history_memory)
        self.assertIn("patterns", app.history_memory[-1])
        self.assertIn("session_id", app.history_memory[-1])
        self.assertIn("planned_task_count", app.history_memory[-1])

    def test_existing_behavioral_agent_signature_still_works(self):
        original_history_file = app.HISTORY_FILE

        with tempfile.NamedTemporaryFile() as history_file:
            app.HISTORY_FILE = history_file.name
            report = app.behavioral_agent("Read paper; draft notes", "Read paper")
            app.HISTORY_FILE = original_history_file

        self.assertIn("Personal Planning Agent - Behavioral Workflow Report", report)


if __name__ == "__main__":
    unittest.main()
