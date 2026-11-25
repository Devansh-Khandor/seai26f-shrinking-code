
"""NASA T7 - Step 4: Flow (NASA-07 and NASA-08)."""

from remember import load_projects, summarize
from guess import get_features, predict


def percentile(values, fraction):
    """Return a simple nearest-rank percentile."""
    ordered = sorted(values)
    index = max(0, int(len(ordered) * fraction + 0.999999) - 1)
    return ordered[index]


def alert_limits(history):
    """Use the upper quartile of observed outcomes as alert limits."""
    return {
        "Effort-": percentile(
            [row["Effort-"] for row in history], 0.75
        ),
        "Defects-": percentile(
            [row["Defects-"] for row in history], 0.75
        ),
    }


def assess_proposal(proposal, history, features):
    """Predict a proposal's outcomes without using its actual goals."""
    limits = alert_limits(history)

    effort = predict(
        proposal, history, features, "Effort-"
    )["estimate"]

    defects = predict(
        proposal, history, features, "Defects-"
    )["estimate"]

    return {
        "row_number": proposal["row_number"],
        "predicted_effort": effort,
        "predicted_defects": defects,
        "effort_alert": effort > limits["Effort-"],
        "defect_alert": defects > limits["Defects-"],
        "effort_limit": limits["Effort-"],
        "defect_limit": limits["Defects-"],
    }


def process_arrivals(history, arrivals, features):
    """Assess each proposal, then reveal its completed outcomes."""
    current = list(history)
    assessments = []

    for project in arrivals:
        # The proposal has known X features but no known Y outcomes.
        proposal = {
            name: value
            for name, value in project.items()
            if name not in ("Kloc+", "Effort-", "Defects-")
        }

        result = assess_proposal(proposal, current, features)
        assessments.append(result)

        # The project is completed after assessment.
        # Only now do its observed goals enter the history.
        current.append(project)

    return current, assessments


def print_summary(title, history):
    """Print updated summary statistics for the goal columns."""
    summary = summarize(history)

    print(f"\n{title}")
    print(f"Historical projects: {len(history)}")
    print(
        f"{'Goal':<12} {'Mean':>12} "
        f"{'Median':>12} {'Max':>12}"
    )

    for goal, values in summary.items():
        print(
            f"{goal:<12}"
            f" {values['mean']:>12.2f}"
            f" {values['median']:>12.2f}"
            f" {values['maximum']:>12.2f}"
        )


def main():
    projects = load_projects()
    features = get_features(projects)

    history = projects[:74]
    arrivals = projects[74:]

    print("NASA T7 - STEP 4: FLOW")
    print(f"Initial completed projects: {len(history)}")
    print(f"Incoming project proposals: {len(arrivals)}")

    print_summary("INITIAL HISTORICAL SUMMARY", history)

    updated_history, assessments = process_arrivals(
        history, arrivals, features
    )

    print("\nNASA-07: INCOMING PROJECT ASSESSMENTS")
    print(
        f"{'Row':>5} {'Effort Est.':>12}"
        f" {'Defect Est.':>13} {'Alerts':>16}"
    )

    effort_alerts = 0
    defect_alerts = 0

    for result in assessments:
        alerts = []

        if result["effort_alert"]:
            alerts.append("Effort")
            effort_alerts += 1

        if result["defect_alert"]:
            alerts.append("Defects")
            defect_alerts += 1

        label = ", ".join(alerts) if alerts else "None"

        print(
            f"{result['row_number']:>5}"
            f" {result['predicted_effort']:>12.2f}"
            f" {result['predicted_defects']:>13.2f}"
            f" {label:>16}"
        )

    print(f"\nHigh-effort alerts: {effort_alerts}")
    print(f"High-defect alerts: {defect_alerts}")

    print_summary(
        "NASA-08: UPDATED HISTORICAL SUMMARY",
        updated_history,
    )

    print("\nStreaming demonstration completed.")
    print(
        f"Final project count: {len(updated_history)}"
    )


if __name__ == "__main__":
    main()
