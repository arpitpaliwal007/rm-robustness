# Trained reward-model audit

## Setup

- Model: `distilroberta-base` reward model trained with a Bradley-Terry pairwise objective
- Training data: 41,538 HH-RLHF preference pairs
- Optimisation: 2,596 steps
- Best validation accuracy: 0.6793
- Evaluation: 8,505 held-out pairs across `helpful-base`, `helpful-online`, `helpful-rejection-sampled`, and `harmless-base`

## Headline results

| Metric | Result |
| --- | ---: |
| Overall held-out accuracy | 0.5643 |
| Length-explained fraction of above-chance accuracy | 0.3638 |
| In-domain accuracy (`helpful-base`) | 0.688 |
| Accuracy on `harmless-base` | 0.351 |
| Reward Spearman correlation with response length | 0.5956 |
| Reward slope | +0.7606 reward SD per 100 tokens |

## Distribution shift

| Evaluation split | Accuracy | Change vs. `helpful-base` |
| --- | ---: | ---: |
| `helpful-base` | 0.688 | — |
| `helpful-online` | 0.560 | -0.129 |
| `helpful-rejection-sampled` | 0.640 | -0.048 |
| `harmless-base` | 0.351 | -0.338 |

The reward distribution shifted by -0.82 reference SD on `harmless-base`, where expected calibration error was 0.316.

## Counterfactual and positional audits

- Direct stance echo: +0.004 reward SD (near-null effect).
- Leading answer versus caveat: +0.175 reward SD.
- Appended disclaimer: +0.139 reward SD; reward increased on 81% of paired rewrites.
- Repeating the final sentence: +0.118 reward SD; reward increased on 86% of paired rewrites.
- Introducing typos: -0.225 reward SD.

These results are from the completed Colab training and evaluation run. They describe a small-model pilot and should not be interpreted as estimates for frontier reward models.
