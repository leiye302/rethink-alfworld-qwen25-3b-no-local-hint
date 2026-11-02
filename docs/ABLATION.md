# Same-start, no-local-hint WebShop3B control

Local publication checks:95 CPU tests passed, one Windows-only platform skip;
the public release API reports matching size/SHA256 metadata for all21 locked
initialization assets. This is not a new GPU or performance result. Clean Linux
CI repeats the native source/tests/environment checks from public resources.

Both arms start from the published Route-SFT1500 model, not independently
retrained SFT models or base:

\[
\theta_0^{\mathrm{main}}=\theta_0^{\mathrm{control}}=\theta_{\mathrm{SFT1500}}.
\]

All trajectories still participate in ordinary native GRPO+OPSD. Teacher is the
current pre-update Actor's no-gradient scoring view, not a frozen initial model.
Conceptually, only its extra local-hint context is removed:

\[
c_{T,i}^{\mathrm{main}}=H_i\oplus S_i
\oplus\mathbf{1}[\text{Actor emitted RETHINK}]\,h_i,
\qquad c_{T,i}^{\mathrm{control}}=H_i\oplus S_i.
\]

This is conceptual concatenation, not a replacement for the pinned native
Teacher builder's exact formatting. Student context and sampled response tokens
remain unchanged. The registered RETHINK token remains free to be generated,
but it activates no local-hint branch here. There is no external API/Analyzer.

This isolates the extra local-hint intervention from a common SFT start. It is
not token removal, Route-SFT removal, OPSD removal, pure GRPO or a base-start
control. It cannot alone establish Route-SFT's causal effect versus base.

## Locks and initialization

`configs/ablation.json` records the immutable main-method commit, complete
recipe and55 shared source/config/data SHA locks. The only allowed recipe delta
is `algorithm.local_hints.enabled: true -> false`. Text locks normalize CRLF to
LF; binary data remain byte-exact. Native patch/losses, frozen SFT data and
fixed140 evaluation identities cannot change silently.

At launch, size/marker/complete-shard checks run and all21 public initialization
assets must match size/SHA. Existing computed weight hashes are reused to avoid
a second12GB hash pass. A base model or newly retrained SFT model is rejected.
The same artifact is downloaded from the declared public main-method release;
no repository-local weights, private server or author's computer is required.
Evaluation may inspect other compatible3B checkpoints; training initialization
remains locked.

Beta5, OPSD coefficient.01, native reference KL.01, optimizer, rollout sampling,
16 tasks ×8 rollouts, history2,15 actions, prompt4096 and response512 remain
unchanged. Train150 updates, validate initial/every10/final on the same140 goals,
save every20/final150 and retain latest2. Keep the same4×A10080GB FSDP4/TP2×DP2
profile, reward, environment and Actor prompt.

## Necessary launcher differences

Default Actor stays `assets/route-sft1500`. Output/experiment name becomes
`runs/sft_start_grpo_opsd`. Credential loading is removed; `--baseline` (which
means base-start in the main repository) is rejected and no paid-API training
flag exists. The ablation contract joins the strict resume identity. Do not
mix main-method/base-start run directories or bypass resume checks.

The main method's thin SDAR patch remains byte-identical for native shared140,
telemetry and generation/row handling. Its existing Teacher conditional takes
`_compute_teacher_log_probs` when local hints are disabled. Tests execute that
actual assignment extracted from the prepared native source, including rows
marked triggered, while making any hint/API call fail the test. Unused shared
hint modules/regression tests remain references, not an active training path.

## Verification boundary

```bash
PYTHONPATH="$PWD/src" python -m rethink_webshop.ablation
```

This standalone contract check needs no GPU or API. CPU tests cover the exact
one-boolean delta, shared sources, public artifact identity, rejected base/paid
flags, no-API/no-CUDA preparation, actual native Teacher branch and missing or
changed initialization files. Public clean CI also checks native WebShop
reset/search/feedback on shared140 seeds using declared public dependencies.

This release is code/configuration, not a newly executed four-A100 training
result. It claims no new reward, success, marker rate or performance. Tiny CPU
fixtures exercise guards; they do not replace the real12GB public initialization
which must be downloaded and fully verified on the training host.
