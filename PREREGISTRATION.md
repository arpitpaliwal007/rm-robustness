# Pre-registration

Written before the T4 run, against a codebase whose probes have been validated on
scripted reward models with known ground truth (`tests/test_probes.py`) but which has
never been pointed at a trained model.

The point of writing this down is that the probe suite produces roughly sixty numbers.
Sixty numbers will always contain something that looks like a finding. Committing to
predictions first is what separates a result from a scan, and a prediction that is
falsified is worth more than one that is confirmed — the RLHF-under-serving-drift pilot
in this project got its strongest result from a corollary dying.

**Reference configuration.** `distilroberta-base`, trained one epoch on `helpful-base`
(≈43.8k pairs), max_length 512, lr 1e-5, batch 8 pairs × grad_accum 2, seed 0. Evaluated
on the test splits of all four subsets. Two additional arms: the same recipe with
`--length-balanced`, and the same recipe trained on `harmless-base`.

---

## What is already known, before any training

Established from the data alone (`results/day0_length_baselines.json`), so these are
constraints on interpretation rather than predictions:

- "Prefer the longer response" scores 0.599 on helpful-base and 0.441 on harmless-base.
- A calibrated length model scores 0.55-0.60 on **every** subset individually, but 0.525
  on the four pooled, because the sign of the effect reverses between them.
- Only 3.5-9.6% of pairs are length-neutral to within 5%.

---

## Predictions

### P1 — Headline magnitude
Reward model accuracy on held-out helpful-base lands in **0.65-0.71**, and the
length-explained fraction lands in **0.40-0.75**.

*Rationale.* Published small-encoder RMs on HH sit in the high 0.60s, and the length
baseline is 0.60. If the model reaches 0.68 while length alone reaches 0.60, most of the
above-chance margin has nowhere else to come from.

*Falsified if* accuracy is below 0.63 (the model failed to train, and nothing downstream
is interpretable) or the fraction is below 0.25 (the model found substantially more than
length, and the headline framing of this study is wrong).

### P2 — The sign flips off-distribution
Trained on helpful-base and evaluated on **harmless-base**, the length-explained fraction
is **negative**: removing the length component *raises* accuracy.

*Rationale.* The model will have learned a positive length coefficient from a corpus
where longer wins. Harmless-base prefers the shorter response. A learned bias pointed the
wrong way is not a shortcut, it is a handicap, and the decomposition should read it as
one. This is the single most diagnostic number in the study, because a correlational
length metric cannot express it: correlation would report "still length-biased" in both
domains and miss that the bias has changed from help to harm.

*Falsified if* the fraction on harmless-base is positive and its CI excludes zero.

### P3 — The coefficient travels
The reward slope per unit log length, in reward-sd units, has the **same sign on all four
subsets**, and its magnitude varies by less than 2× between them.

*Rationale.* Nothing in BT training tells the model that the length-preference direction
is domain-conditional. It should carry one coefficient everywhere. Together with P2 this
is the mechanism: the coefficient does not adapt, so it helps in two subsets and hurts in
two.

*Falsified if* the sign flips on any subset — which would mean the model has learned a
domain-conditional length rule from context alone, a considerably more interesting result
than the one predicted.

### P4 — Sycophancy is real but small
The stance-flip effect net of placebo is **positive, between 0.05 and 0.20 reward sd**,
and the placebo floor is **below 0.03 reward sd** in absolute value.

*Rationale.* HH annotators chose the more helpful response, not the more agreeable one,
so agreement is not directly rewarded. But responses that open by validating the user
correlate with responses rated helpful, and the model has no way to separate the two. The
2×2 design removes every confound except that correlation, so what survives should be
small and positive.

*Falsified if* the net effect's CI includes zero (no measurable sycophancy at this model
scale) or if the placebo floor exceeds half the stance-flip effect (the instrument is
measuring a generic context effect and the primary number is not about agreement).

### P5 — Flattery beats critique; capitulation is a coin flip
Praise over critique of the user's own work: **positive, 0.10-0.40 reward sd**, and still
positive at equal length. Abandoning a correct answer under pushback: **CI includes
zero**.

*Rationale.* These are different mechanisms. Flattery is warmth, which HH raters reward.
Capitulation is warmth applied to being wrong, and the raters had no reason to prefer
either version; the model is unlikely to have learned a signal that was not in the labels.

*Falsified if* capitulation is significantly positive — which would be the more alarming
result and worth a project of its own.

### P6 — Positional sensitivity without positional bias
On the order-invariant swap items, the **mean absolute** effect is **0.05-0.25 reward
sd** while the **signed** effect's CI **includes zero**. On the answer-vs-caveat items the
signed effect is positive (answer first scores higher). Sentence reversal shows a mean
absolute shift above 0.05 sd, confirming the model is not order-blind.

*Rationale.* A transformer with position embeddings will not score two orderings
identically, but there is no reason for the noise to have a preferred direction on
genuinely interchangeable content. Answer-first is different: HH annotators saw hedge-first
responses as evasive.

*Falsified if* the signed effect on order-invariant items is significantly non-zero (a
real primacy or recency bias, which would be a finding), or if sentence reversal comes
back near zero, in which case the swap null is uninformative and P6 is untestable rather
than confirmed.

### P7 — The model grades the opening
On **helpful-online** (mean response 106 words, so truncation actually bites), the first
**32 tokens recover at least 60%** of the model's above-chance accuracy, and the first 64
recover at least 80%.

*Rationale.* BT training gives one scalar per response with no pressure to integrate over
the whole text, and the strongest surface cues — an opener, a refusal, a hedge — are all
at the front. Stated on helpful-online rather than helpful-base deliberately: on
helpful-base the mean response is 44 words, so a 64-token prefix is most of the response
and any result would be trivial.

*Falsified if* 32 tokens recover under 40%.

### P8 — Constraints go out of view
The compliant-minus-violating gap at 3 filler turn pairs retains between **0.3 and 0.8**
of its value at 0.

*Rationale.* The constraint stays well inside the 512-token window, so this is not
truncation — it is attention. Some decay is expected from dilution; total collapse would
mean the model effectively conditions on the last turn only.

*Falsified if* retention exceeds 0.95 (no depth effect at all) or is below 0.1 (the model
is a last-turn scorer, which would make most multi-turn RLHF evaluation suspect).

### P9 — Free reward from surface habits
`appended_disclaimer`, `hedge_opener` and `repeat_last_sentence` all produce positive
reward shifts. `lowercase`, which changes no content and adds no tokens, produces an
absolute shift **above 0.05 reward sd** and a decision flip rate **between 2% and 12%**.

*Rationale.* The length-changing rewrites are P3 restated, and should be predictable from
the reward slope. Lowercasing is the interesting one: it is exactly the invariance a
reward function ought to have, and subword tokenizers do not have it.

*Falsified if* lowercase flips no decisions at all, which would be a genuine and
surprising robustness result.

### P10 — Calibration degrades faster than accuracy
In-domain ECE below 0.05; ECE on harmless-base at least **2× the in-domain value**, and
the reward mean shift between the two domains at least **0.3 reference sd**.

*Rationale.* BT training calibrates the margin distribution to the training domain. Both
the scale and the location of the reward move under shift, and neither is visible in an
accuracy number — which is the practical warning this probe exists to give.

### P11 — The control arm survives
The `--length-balanced` model, evaluated on held-out helpful-base, scores between
**0.55 and 0.62**.

*Rationale.* Above chance, because there is real preference signal in HH beyond length.
Below the unbalanced model, because a genuine cue has been removed and the training set
is smaller. This is the cleanest single test of whether the headline fraction is telling
the truth: if P1 says length explains 60% of the skill, an RM denied length should keep
roughly the other 40% of the margin, i.e. land near 0.57.

*Falsified if* it scores at or below 0.52 — length was not a shortcut, it was the entire
signal, and the honest conclusion is that a small RM cannot learn HH preferences at all —
or above 0.65, in which case the balanced resampling did not remove what it claims to.

---

## Analysis decisions fixed in advance

- Length in **model tokens**, post-truncation, as the primary unit; chars reported
  alongside.
- The length-explained fraction is computed with **2-fold cross-fitting** and is
  suppressed whenever the accuracy CI includes chance.
- All probe-set intervals are **cluster-bootstrapped by topic**, never by item.
- Calibration is **symmetrised**.
- Every length metric is reported a second time with truncated pairs excluded, and the
  truncated-excluded version is the one quoted if the two disagree by more than 0.05.
- `n_boot = 2000`, `seed = 0`, α = 0.05 throughout.
- Subgroup results are reported for all four subsets whether or not they are interesting.
  No subset is dropped after seeing it.

## What would make the whole study uninteresting

If the trained model turns out to be **worse** than the calibrated length baseline on
every subset, then there is no reward model here to probe, only a length detector with
extra steps, and the sycophancy and position results would be measuring the properties of
noise. The first check after training is therefore `acc_rm` against
`acc_length_logistic_oof`, per subset. If the gap is under 2 points anywhere, say so in
the report before quoting anything else.
