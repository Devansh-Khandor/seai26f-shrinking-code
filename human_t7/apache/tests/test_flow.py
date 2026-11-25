"""Behavioral tests for Apache T7 Flow."""

import math
import sys
import unittest
from pathlib import Path

APACHE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(APACHE_DIR))

from flow import (
    StreamMonitor,
    historical_quantile,
    simulate_stream,
    validate_final_state,
)
from guess import train_test_split
from remember import GOAL, load_configurations


def make_row(number, performance):
    return {
        "row_number": number,
        "base": 1,
        "keepAlive": number % 2,
        GOAL: float(performance),
    }


class TestFlow(unittest.TestCase):

    def setUp(self):
        self.history = [
            make_row(1, 10),
            make_row(2, 20),
            make_row(3, 30),
            make_row(4, 40),
        ]

    def test_historical_quantile(self):
        self.assertEqual(
            historical_quantile([10, 20, 30, 40]), 40
        )
        self.assertEqual(
            historical_quantile([10, 20, 30, 40], 0.5), 20
        )

    def test_empty_history_rejected(self):
        with self.assertRaises(ValueError):
            StreamMonitor([])

    def test_running_summary(self):
        monitor = StreamMonitor(self.history)
        summary = monitor.snapshot()

        self.assertEqual(summary["count"], 4)
        self.assertEqual(summary["minimum"], 10)
        self.assertEqual(summary["maximum"], 40)
        self.assertEqual(summary["median"], 25)
        self.assertEqual(summary["mean"], 25)

    def test_alert_uses_only_prior_history(self):
        monitor = StreamMonitor(self.history)

        # Prior 90th percentile = 40.
        event = monitor.process_arrival(make_row(5, 100))

        self.assertTrue(event["alert"])
        self.assertEqual(event["historical_p90"], 40)
        self.assertEqual(event["prior_count"], 4)

        # The previously arrived 100 is now historical.
        event2 = monitor.process_arrival(make_row(6, 90))

        self.assertFalse(event2["alert"])
        self.assertEqual(event2["historical_p90"], 100)
        self.assertEqual(event2["prior_count"], 5)

    def test_threshold_is_strict(self):
        monitor = StreamMonitor(self.history)
        event = monitor.process_arrival(make_row(5, 40))

        self.assertFalse(event["alert"])

    def test_arrival_updates_summary(self):
        monitor = StreamMonitor(self.history)
        event = monitor.process_arrival(make_row(5, 100))

        summary = event["updated_summary"]
        self.assertEqual(summary["count"], 5)
        self.assertEqual(summary["maximum"], 100)
        self.assertEqual(summary["median"], 30)
        self.assertEqual(summary["mean"], 40)

    def test_shortlist_updates(self):
        monitor = StreamMonitor(
            self.history, shortlist_size=3
        )

        monitor.process_arrival(make_row(5, 5))

        self.assertEqual(
            [r["row_number"] for r in monitor.top_configurations()],
            [5, 1, 2],
        )

    def test_duplicate_arrival_rejected(self):
        monitor = StreamMonitor(self.history)

        with self.assertRaises(ValueError):
            monitor.process_arrival(make_row(1, 999))

        self.assertEqual(monitor.snapshot()["count"], 4)

    def test_nonfinite_performance_rejected(self):
        monitor = StreamMonitor(self.history)

        with self.assertRaises(ValueError):
            monitor.process_arrival(make_row(5, math.nan))

        self.assertEqual(monitor.snapshot()["count"], 4)

    def test_input_not_mutated(self):
        original = [row.copy() for row in self.history]
        monitor = StreamMonitor(self.history)

        monitor.process_arrival(make_row(5, 50))
        self.assertEqual(self.history, original)

    def test_final_batch_consistency(self):
        arrivals = [
            make_row(5, 5),
            make_row(6, 100),
        ]
        monitor, events = simulate_stream(
            self.history, arrivals
        )

        self.assertEqual(len(events), 2)
        self.assertTrue(
            validate_final_state(
                monitor, self.history + arrivals
            )
        )

    def test_real_dataset_full_reconciliation(self):
        configurations = load_configurations()
        history, arrivals = train_test_split(configurations)

        self.assertEqual(len(history), 154)
        self.assertEqual(len(arrivals), 38)

        monitor, events = simulate_stream(
            history, arrivals
        )

        self.assertEqual(len(events), 38)
        self.assertEqual(monitor.snapshot()["count"], 192)
        self.assertTrue(
            validate_final_state(monitor, configurations)
        )

    def test_every_arrival_sees_correct_prior_count(self):
        configurations = load_configurations()
        history, arrivals = train_test_split(configurations)

        _, events = simulate_stream(history, arrivals)

        self.assertEqual(
            [event["prior_count"] for event in events],
            list(range(154, 192)),
        )


if __name__ == "__main__":
    unittest.main()
