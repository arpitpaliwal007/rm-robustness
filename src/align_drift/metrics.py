from __future__ import annotations

from collections import defaultdict


def accuracy_by_shift(rows: list[dict], scorer: str) -> dict[str, float]:
    """Accuracy grouped by PPO policy-shift label."""
    correct, total = defaultdict(int), defaultdict(int)
    for row in rows:
        shift = str(row["shift"])
        total[shift] += 1
        correct[shift] += int(row[scorer] == row["chosen"])
    return {shift: correct[shift] / total[shift] for shift in sorted(total)}
