# ALFWorld3B: same-start no-local-hint control

## Scientific boundary

Start from the exact full-correction Route-SFT2000 checkpoint used by Full RETHINK.
Retain the tokenizer/marker, prompts, action parser, rewards, static skills,
rollout budget, Teacher scoring, reference anchor and native losses.
Disable only Teacher-local hint augmentation in Stage2.
This is not a no-RETHINK, no-SFT, pure-GRPO, or external-router experiment.

The unchanged supervised initialization has

```math
\mathcal L_{\mathrm{SFT}}
=\mathcal L_{\mathrm{suffix\text{-}CE}}+0.1\mathcal L_{\mathrm{marker\text{-}BCE}}.
```

H/R tokens are masked from CE. Good keeps its original natural action;
Bad learns the marker and full corrected action/EOS. BCE uses the full-vocabulary
marker probability at the last prefix position, not the marker's raw-logit sigmoid.
Loss core/data files are checked against the immutable full3B recipe.

## Stage2 invariants

150 updates; batch16 tasks ×8 rollouts; Adam learning rate1e-6;
SDAR gate beta5; OPSD.01; reference KL.01; entropy.001.
Current Actor is the no-grad Teacher; reference is the shared initial Actor.
Teacher sees native static ALFWorld skills, Student never does.
No decision Analyzer or episode Analyzer runs.
Native parser, training prompts, reward and loss aggregation are unchanged.
One80/96GB GPU, TP1, native token-mean/microbatch32 configuration.
Changing hardware/microbatch requires a matched adaptation for Full as well.

Train games3553 are disjoint from140 valid_seen games. Validation is initial,
every10 updates and final; saves every20 and final, latest2 retained.
All evaluations use identical ordered tasks/seeds, history2, max50 actions,
prompt2048, response512, temperature.4 and sampling; no hint or Teacher.

## Resource and license

README installation prepares public pinned SDAR plus the bundled frozen dataset.
The exact common checkpoint is a public GitHub Release. Download/import checks
each filename, size and SHA. Original model/tokenizer files are unchanged;
the private optimizer is excluded. Qwen Research License and modification notice
are included; code retains its separate Apache license.
No author's disk, login, API balance or private path is a dependency.

If intentionally creating a NEW common initializer instead, first download
the pinned public base using `setup.py --download-base --accept-model-license`,
then run `train.sh sft`. This is full-correction SFT,1epoch2000 updates, global
batch8(5Good/3Bad), micro1, LR2e-6, AdamW(.9,.95), weight decay.01, clip1,
seed0, max8192, truncation error, no reasoning KL, BCE.1. The LR horizon
remains4000 with400 warmup steps, matching the original first2000 steps.
Do not independently rerun SFT for the other arm: share these exact saved weights.
This alternative does not claim bitwise identity to the historical released model.

## Runtime safeguards

Source/config/data bindings prevent silently reusing another run's checkpoints.
Launch locks prevent duplicate controllers. Incomplete model/state saves are
not accepted as finished stages. No blind restart or automatic deletion.
`OPSD_PYTHON=python` resolves through PATH; relative paths resolve at repo root.
Comparison checks canonical task identity plus only verified legacy raw hashes.
Unknown task hashes, changed seeds/order/game contents, incomplete evaluations,
wrong update/split/count, or non-Actor evaluation are rejected.

## Interpretation

Compare Full versus no-hint only when both use the same initial artifact,
training/validation budget and hardware profile. Report final seen140 success,
marker frequency, reasoning completeness and cost separately. No new experimental
score is established by packaging/CPU tests. Do not treat inherited WebShop
history as an active run or claim.
