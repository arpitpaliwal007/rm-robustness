from __future__ import annotations

import argparse
import json
from pathlib import Path

from .metrics import accuracy_by_shift


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate reward models and LLM judges across PPO policy shifts.")
    parser.add_argument("--predictions", type=Path, required=True); parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = [json.loads(line) for line in args.predictions.read_text().splitlines() if line]
    report = {"reward_model_accuracy": accuracy_by_shift(rows, "reward_model_choice"), "llm_judge_accuracy": accuracy_by_shift(rows, "llm_judge_choice"), "examples": len(rows)}
    args.output.parent.mkdir(parents=True, exist_ok=True); args.output.write_text(json.dumps(report, indent=2)); print(json.dumps(report, indent=2))


if __name__ == "__main__": main()
