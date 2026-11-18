"""Apache T7, Step 4: Flow (APACHE-07 and APACHE-08)."""

import bisect
import math
import statistics

from remember import (
    GOAL,
    load_configurations,
    rank_configurations,
    summarize,
)
from guess import train_test_split

ALERT_QUANTILE = 0.90
SHORTLIST_SIZE = 5


def historical_quantile(sorted_values, quantile=ALERT_QUANTILE):
    """Return a nearest-rank quantile from sorted historical values."""
    if not sorted_values:
        raise ValueError("Historical values cannot be empty")
    if not 0 < quantile <= 1:
        raise ValueError("Quantile must be in (0, 1]")

    index = math.ceil(quantile * len(sorted_values)) - 1
    return sorted_values[index]


class StreamMonitor:
    """Maintain running statistics and a best-configuration shortlist."""

    def __init__(self, historical_rows, shortlist_size=SHORTLIST_SIZE):
        if not historical_rows:
            raise ValueError("Initial history cannot be empty")
        if shortlist_size < 1:
            raise ValueError("Shortlist size must be positive")

        self.shortlist_size = shortlist_size
        self.rows = []
        self.seen_ids = set()
        self.sorted_values = []
        self.shortlist = []
        self.running_sum = 0.0

        for row in historical_rows:
            self._insert(row)

    def _insert(self, row):
        """Update all state without rebuilding from the dataset."""
        row_id = row["row_number"]
        performance = float(row[GOAL])

        if row_id in self.seen_ids:
            raise ValueError(f"Duplicate row_number: {row_id}")
        if not math.isfinite(performance):
            raise ValueError("Performance must be finite")

        saved = row.copy()
        saved[GOAL] = performance

        self.seen_ids.add(row_id)
        self.rows.append(saved)

        bisect.insort(self.sorted_values, performance)
        self.running_sum += performance

        # Lower performance is better. CSV row breaks ties.
        entry = (performance, row_id, saved)
        if (
            len(self.shortlist) < self.shortlist_size
            or (performance, row_id) <
               self.shortlist[-1][:2]
        ):
            bisect.insort(self.shortlist, entry)
            del self.shortlist[self.shortlist_size:]

    def snapshot(self):
        """Return currently available running statistics."""
        return {
            "count": len(self.rows),
            "minimum": self.sorted_values[0],
            "median": statistics.median(self.sorted_values),
            "mean": self.running_sum / len(self.rows),
            "maximum": self.sorted_values[-1],
        }

    def top_configurations(self):
        """Return the current lowest-performance configurations."""
        return [entry[2].copy() for entry in self.shortlist]

    def process_arrival(self, row):
        """Evaluate an arrival against past history, then update state."""
        row_id = row["row_number"]
        performance = float(row[GOAL])

        if row_id in self.seen_ids:
            raise ValueError(f"Duplicate arrival: {row_id}")
        if not math.isfinite(performance):
            raise ValueError("Performance must be finite")

        # CRITICAL: compute the alert using only previous records.
        prior_count = len(self.rows)
        threshold = historical_quantile(self.sorted_values)
        poor_performance = performance > threshold

        # Only now make the arriving record available to the system.
        self._insert(row)

        return {
            "row_number": row_id,
            "performance": performance,
            "prior_count": prior_count,
            "historical_p90": threshold,
            "alert": poor_performance,
            "updated_summary": self.snapshot(),
            "updated_top5": [
                item["row_number"]
                for item in self.top_configurations()
            ],
        }


def simulate_stream(history, arrivals):
    """Process every arrival sequentially without looking ahead."""
    monitor = StreamMonitor(history)
    events = []

    for row in arrivals:
        events.append(monitor.process_arrival(row))

    return monitor, events


def validate_final_state(monitor, all_rows):
    """Check streaming results against independent batch summaries."""
    batch = summarize(all_rows)
    streamed = monitor.snapshot()

    if streamed["count"] != batch["count"]:
        return False

    for field in ("minimum", "median", "mean", "maximum"):
        if not math.isclose(
            streamed[field],
            batch[field],
            rel_tol=1e-10,
            abs_tol=1e-8,
        ):
            return False

    batch_top = [
        row["row_number"]
        for row in rank_configurations(all_rows)[:SHORTLIST_SIZE]
    ]
    stream_top = [
        row["row_number"]
        for row in monitor.top_configurations()
    ]

    return batch_top == stream_top


def main():
    configurations = load_configurations()
    history, arrivals = train_test_split(configurations)

    print("APACHE T7 — STEP 4: FLOW")
    print(f"\nInitial history: {len(history)}")
    print(f"Incoming benchmarks: {len(arrivals)}")
    print("Alert rule: performance > prior historical 90th percentile")
    print("Lower performance is better.")

    monitor = StreamMonitor(history)

    print("\nAPACHE-07: SEQUENTIAL BENCHMARK ALERTS")
    print(
        f"{'Arrival':>7} {'Row':>5} {'Measured':>10}"
        f" {'Prior P90':>10} {'Alert':>7}"
    )

    alerts = 0

    for index, row in enumerate(arrivals, start=1):
        event = monitor.process_arrival(row)
        alerts += event["alert"]

        print(
            f"{index:>7}"
            f" {event['row_number']:>5}"
            f" {event['performance']:>10.2f}"
            f" {event['historical_p90']:>10.2f}"
            f" {'YES' if event['alert'] else 'NO':>7}"
        )

    print(f"\nArrivals processed: {len(arrivals)}")
    print(f"Unusually poor performance alerts: {alerts}")

    print("\nAPACHE-08: UPDATED PERFORMANCE SUMMARY")
    for name, value in monitor.snapshot().items():
        print(f"{name:<12}: {value:.2f}")

    print("\nAPACHE-08: UPDATED TOP FIVE CONFIGURATIONS")
    print(f"{'Row':>5} {'Performance':>13}")

    for row in monitor.top_configurations():
        print(f"{row['row_number']:>5} {row[GOAL]:>13.2f}")

    valid = validate_final_state(monitor, configurations)

    print("\nFINAL STATE VERIFICATION")
    print(f"Expected inventory: {len(configurations)}")
    print(f"Streaming inventory: {len(monitor.rows)}")
    print(f"Matches complete batch results: {valid}")

    if not valid:
        raise AssertionError(
            "Streaming state differs from full-data batch results"
        )

    print(
        "\nAlerts were evaluated before incorporating each arrival."
        "\nThe stream uses a simulated arrival order, not real timestamps."
    )


if __name__ == "__main__":
    main()
