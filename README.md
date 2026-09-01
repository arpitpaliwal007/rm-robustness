# Reward Models and LLM Judges under Policy Shift

Evaluation pipeline for comparing reward models with LLM judges as PPO policies move away from their initial response distribution.

## Problem

Reward models are normally evaluated on static preference datasets. After PPO changes a policy, its responses can differ from the data used to train the reward model. This project measures whether reward-model accuracy remains reliable under those policy shifts, and compares it with an LLM judge.

## Setup

- Train or load Qwen3 policy checkpoints at several PPO stages.
- Generate responses from each stage for prompts from RewardBench, JudgeBench, and Skywork-80K.
- Score the same response pairs with a 3B reward model and an LLM judge.
- Report accuracy by dataset and policy-shift level.

## Reported result

Across increasing policy shifts, the recorded reward-model accuracy fell from **72% to 50%**. The LLM judge remained between **80% and 88%**. These reference values are stored in `results/reference_shift_report.json`; reproduce them with the same policy checkpoints and evaluation split before using them as a new run.

## Run

```bash
pip install -r requirements.txt
python -m align_drift.evaluate --predictions data/pairs.jsonl --output results/run.json
```

The input JSONL uses: `prompt`, `response_a`, `response_b`, `chosen`, and `shift`.

## Test

```bash
python -m unittest discover -s tests -v
```
