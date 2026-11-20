
from pathlib import Path
import subprocess
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]

SKILLS = [
    "remember",
    "guess",
    "certify",
    "flow",
    "blame",
    "justify",
    "choose",
    "fix",
    "spot",
    "race",
]


class HumanT7CarsTests(unittest.TestCase):
    def test_all_skills_execute_successfully(self):
        for skill in SKILLS:
            with self.subTest(skill=skill):
                result = subprocess.run(
                    [sys.executable, "-m", f"human_t7.cars.{skill}"],
                    cwd=ROOT,
                    capture_output=True,
                    text=True,
                )

                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertTrue(result.stdout.strip())

    def test_guess_outputs_expected_estimates(self):
        result = subprocess.run(
            [sys.executable, "-m", "human_t7.cars.guess"],
            cwd=ROOT,
            capture_output=True,
            text=True,
        )

        self.assertEqual(result.returncode, 0, result.stderr)

        expected = [
            "Historical records: 398",
            "Training records: 318",
            "Testing records: 80",
            "CARS-03: Fuel efficiency estimation",
            "CARS-04: Vehicle weight estimation",
            "MAE:          3.30",
            "Baseline MAE: 7.81",
            "MAE:          227.34",
            "Baseline MAE: 758.53",
            "Vehicle ID: 4",
            "Fuel efficiency: 16.00 MPG",
            "Vehicle weight: 3885.20 lbs",
            "Neighbor range:",
            "Neighbor StdDev:",
        ]

        for text in expected:
            with self.subTest(expected=text):
                self.assertIn(text, result.stdout)

    def test_certify_outputs_expected_comparisons(self):
        result = subprocess.run(
            [sys.executable, "-m", "human_t7.cars.certify"],
            cwd=ROOT,
            capture_output=True,
            text=True,
        )

        self.assertEqual(result.returncode, 0, result.stderr)

        expected = [
            "Historical records: 398",
            "CAR-05: Fuel efficiency by cylinder count",
            "3 cylinders:",
            "Vehicles: 4",
            "Average MPG: 20.00",
            "4 cylinders:",
            "Vehicles: 204",
            "Average MPG: 29.51",
            "6 cylinders:",
            "Vehicles: 87",
            "8 cylinders:",
            "CAR-06: Overall performance by model-year period",
            "1970-1973:",
            "Vehicles: 125",
            "Average performance score: 0.6506",
            "1974-1977:",
            "Vehicles: 119",
            "1978-1982:",
            "Vehicles: 154",
            "Model-year period comparisons:",
            "Within thresholds: False",
            "Thresholds exceeded:",
        ]

        for text in expected:
            with self.subTest(expected=text):
                self.assertIn(text, result.stdout)

    def test_flow_updates_inventory_incrementally(self):
        result = subprocess.run(
            [sys.executable, "-m", "human_t7.cars.flow"],
            cwd=ROOT,
            capture_output=True,
            text=True,
        )

        self.assertEqual(result.returncode, 0, result.stderr)

        expected = [
            "Initial inventory: 50 vehicles",
            "CAR-07: Assess each arriving vehicle",
            "Previous inventory: 50",
            "Previous inventory: 51",
            "Previous inventory: 52",
            "Previous inventory: 53",
            "Previous inventory: 54",
            "CAR-08: Updated inventory",
            "Accepted new vehicles: 5",
            "Current inventory size: 55",
            "Updated summaries:",
            "Mpg+: Count=55",
            "Lbs-: Count=55",
            "Current top-five vehicles:",
        ]

        for text in expected:
            with self.subTest(expected=text):
                self.assertIn(text, result.stdout)

    def test_blame_identifies_poor_performance_characteristics(self):
        result = subprocess.run(
            [sys.executable, "-m", "human_t7.cars.blame"],
            cwd=ROOT,
            capture_output=True,
            text=True,
        )

        self.assertEqual(result.returncode, 0, result.stderr)

        expected = [
            "Historical records: 398",
            "CAR-09: Characteristics associated with poor MPG",
            "Volume",
            "Cliff's delta: 0.996",
            "Clndrs",
            "Cliff's delta: 0.994",
            "origin",
            "Some nonzero category counts are below five",
            "CAR-10: Characteristics associated with poor overall performance",
            "Cliff's delta: 1.000",
            "Cliff's delta: 0.616",
            "Separation: 0.747",
        ]

        for text in expected:
            with self.subTest(expected=text):
                self.assertIn(text, result.stdout)

    def test_justify_explains_recommendations_and_tradeoffs(self):
        result = subprocess.run(
            [sys.executable, "-m", "human_t7.cars.justify"],
            cwd=ROOT,
            capture_output=True,
            text=True,
        )

        self.assertEqual(result.returncode, 0, result.stderr)

        expected = [
            "CAR-11: Recommendation explanation",
            "Recommended vehicle:",
            "Vehicle ID: 395",
            "Goal contributions:",
            "Lbs-:",
            "Acc+:",
            "Mpg+:",
            "Three most similar historical vehicles:",
            "Vehicle 368:",
            "Vehicle 361:",
            "Vehicle 347:",
            "CAR-12: Strengths and trade-offs",
            "Vehicle A:",
            "Vehicle B:",
            "Goals favoring A: ('Acc+',)",
            "Goals favoring B: ('Lbs-',)",
            "Equal goals: ('Mpg+',)",
            "Overall winner: Vehicle 395",
            "lower combined distance-to-heaven score",
        ]

        for text in expected:
            with self.subTest(expected=text):
                self.assertIn(text, result.stdout)

    def test_choose_selects_and_prioritizes_candidates(self):
        result = subprocess.run(
            [sys.executable, "-m", "human_t7.cars.choose"],
            cwd=ROOT,
            capture_output=True,
            text=True,
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        output = result.stdout

        # CAR-13: selection checkpoints
        self.assertIn("After 5 inspections:", output)
        self.assertIn("After 10 inspections:", output)
        self.assertIn("After 20 inspections:", output)
        self.assertIn("After 30 inspections:", output)

        self.assertIn("Best vehicle: 390", output)
        self.assertIn("Actual score: 0.2381", output)

        # Offline benchmark
        self.assertIn("Best candidate: 395", output)
        self.assertIn("Best possible score: 0.1708", output)

        # CAR-14: inspection priority
        self.assertIn("1. Vehicle 291:", output)
        self.assertIn("30. Vehicle 308:", output)
        self.assertIn("Predicted=", output)
        self.assertIn("Observed=", output)

        # Inspection budget and oracle usage
        self.assertIn("Inspections performed: 30", output)
        self.assertIn("Predictions made: 13095", output)
        self.assertIn("Oracle inspections: 30", output)

    def test_fix_finds_better_alternatives_and_repairs_targets(self):
        result = subprocess.run(
            [sys.executable, "-m", "human_t7.cars.fix"],
            cwd=ROOT,
            capture_output=True,
            text=True,
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        output = result.stdout

        # CAR-15: Better-performing historical alternatives
        self.assertIn("Historical records: 398", output)
        self.assertIn("Source vehicle: 15", output)
        self.assertIn("Source performance score: 0.9275", output)

        self.assertIn("CAR-15: Better-performing historical alternatives", output)
        self.assertIn("1 Vehicle 40", output)
        self.assertIn("2 Vehicle 14", output)
        self.assertIn("Pareto dominates source: True", output)
        self.assertIn("Volume: 400.0 -> 383.0", output)
        self.assertIn("Model: 71.0 -> 70.0", output)

        # CAR-16: Closest alternative meeting more targets
        self.assertIn("CAR-16: Closest alternative meeting more targets", output)
        self.assertIn("Lbs-: at most 2803.50", output)
        self.assertIn("Acc+: at least 15.50", output)
        self.assertIn("Mpg+: at least 20.00", output)

        self.assertIn("Alternative vehicle: 58", output)
        self.assertIn("Targets satisfied by source: 0", output)
        self.assertIn("Targets satisfied by alternative: 1", output)
        self.assertIn("Number of characteristic changes: 1", output)
        self.assertIn("Newly satisfied targets:", output)
        self.assertIn("Mpg+", output)

    
    def test_spot_flags_unusual_characteristics_and_unexpected_outcomes(self):
        result = subprocess.run(
            [sys.executable, "-m", "human_t7.cars.spot"],
            cwd=ROOT,
            capture_output=True,
            text=True,
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        output = result.stdout

        # CAR-17: Unusual vehicle characteristics
        self.assertIn("Historical records: 398", output)
        self.assertIn("CAR-17: Unusual vehicle characteristics", output)

        for vehicle in (155, 142, 193, 243, 226):
            with self.subTest(vehicle=vehicle):
                self.assertIn(f"Vehicle {vehicle}: FLAGGED", output)

        self.assertIn("Mean neighbor distance: 0.1841", output)
        self.assertIn("Top 5 mean nearest-neighbor X distances; review only", output)
        self.assertIn("Nearest neighbors:", output)

        # CAR-18: Unexpected vehicle outcomes
        self.assertIn("CAR-18: Unexpected vehicle outcomes", output)
        self.assertIn("Training vehicles: 318", output)
        self.assertIn("Testing vehicles: 80", output)

        self.assertIn("Mpg+: Largest prediction errors", output)
        self.assertIn("Vehicle 182: FLAGGED", output)
        self.assertIn("Absolute error: 12.00", output)
        self.assertIn("Signed error: -12.00", output)

        self.assertIn("Lbs-: Largest prediction errors", output)
        self.assertIn("Vehicle 141: FLAGGED", output)
        self.assertIn("Absolute error: 875.40", output)
        self.assertIn("Signed error: 875.40", output)

        self.assertIn("Top 5 absolute held-out Mpg+ residuals; review only", output)
        self.assertIn("Top 5 absolute held-out Lbs- residuals; review only", output)

    
    def test_race_compares_selection_and_prediction_baselines(self):
        result = subprocess.run(
            [sys.executable, "-m", "human_t7.cars.race"],
            cwd=ROOT,
            capture_output=True,
            text=True,
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        output = result.stdout

        # CAR-19: Model selection vs. random selection
        self.assertIn("CAR-19: Model selection vs. random selection", output)

        for budget in (10, 20, 30):
            with self.subTest(budget=budget):
                self.assertIn(f"Inspection budget: {budget}", output)

        self.assertIn("Model best score: 0.1675", output)
        self.assertIn("Random mean score: 0.2458", output)
        self.assertIn("Random median score: 0.2583", output)
        self.assertIn("Random trials matching or beating model: 20.0%", output)
        self.assertIn("Valid random trials: 30/30", output)

        # CAR-20: kNN vs. historical-average prediction
        self.assertIn("CAR-20: kNN vs. historical-average prediction", output)
        self.assertIn("Training vehicles: 318", output)
        self.assertIn("Testing vehicles: 80", output)

        self.assertIn("Mpg+: Prediction comparison", output)
        self.assertIn("kNN MAE: 3.30", output)
        self.assertIn("Historical-average MAE: 7.81", output)
        self.assertIn("Percentage improvement: 57.73%", output)
        self.assertIn("kNN wins: 64", output)

        self.assertIn("Lbs-: Prediction comparison", output)
        self.assertIn("kNN MAE: 227.34", output)
        self.assertIn("Historical-average MAE: 758.53", output)
        self.assertIn("Percentage improvement: 70.03%", output)
        self.assertIn("kNN wins: 66", output)


    def test_remember_summarizes_inventory_and_ranks_attractive_vehicles(self):
        result = subprocess.run(
            [sys.executable, "-m", "human_t7.cars.remember"],
            cwd=ROOT,
            capture_output=True,
            text=True,
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        output = result.stdout

        # CAR-01: Historical vehicle records summary
        self.assertIn("Loaded 398 historical vehicle records", output)
        self.assertIn("CARS-01: Historical vehicle records summary", output)

        expected_summary = [
            "Lbs-",
            "Mpg+",
            "Acc+",
            "Count:  398",
            "Mean:   2970.42",
            "Median: 2803.50",
            "Best:   1613.0",
            "Worst:  5140.0",
            "Mean:   23.84",
            "Median: 20.00",
            "Mean:   15.57",
            "Median: 15.50",
        ]

        for text in expected_summary:
            with self.subTest(expected=text):
                self.assertIn(text, output)

        # CAR-02: Attractive overall combinations
        self.assertIn("CARS-02: 10 Most attractive overall combination", output)
        self.assertIn("Vehicle 395: Weight=2130.0", output)
        self.assertIn("Acceleration=24.6, MPG=40.0, Score=0.167", output)
        self.assertIn("Vehicle 394:", output)
        self.assertIn("Vehicle 393:", output)
        self.assertIn("Vehicle 396:", output)
        self.assertIn("Vehicle 389:", output)

if __name__ == "__main__":
    unittest.main()
