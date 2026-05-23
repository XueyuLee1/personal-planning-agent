import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path


APP_PATH = Path(__file__).resolve().parents[1] / "app(2).py"
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


if __name__ == "__main__":
    unittest.main()
