# ALFWorld Qwen2.5-3B — no local hint ablation

Built with Qwen. **Full Route-SFT2000 → GRPO+OPSD150, without local hints.**
This repository corrects an earlier WebShop misconfiguration. It is ALFWorld,
not WebShop, not base-start, and not the trigger-only SFT ablation.

| Component | Full method | This ablation |
|---|---|---|
| Actor / reference initialization | Full Route-SFT step2000 | Exact same saved files |
| SFT Good / Bad | Original action / RETHINK + correction | Unchanged |
| Teacher | Current Actor weights, no-grad scoring | Unchanged |
| Static SDAR ALFWorld skills | Teacher-only | Unchanged |
| Extra hint at Actor-generated RETHINK | Teacher-only local hint | **Disabled, no API/Analyzer** |
| RETHINK token | Free Actor generation | Retained; neither forced nor suppressed |
| GRPO, OPSD, reference KL, entropy | Native SDAR | Unchanged |
| Gate beta / OPSD / KL / entropy coefficients | 5 / .01 / .01 / .001 | Unchanged |
| Updates / save / validation | 150 / every20 / every10 | Unchanged |
| Final evaluation | Fixed ALFWorld valid_seen140 | Same ordered tasks, seeds and decoding |

The only Stage-2 algorithm delta is `algorithm.local_hints.enabled: true -> false`.
All decisions, including RETHINK decisions, use native
`_compute_teacher_log_probs` with ordinary static-skill context.
GRPO still covers the whole trajectory. The Teacher is **not frozen**;
the reference model is anchored at the shared RL initialization.

## Installation

Linux, CUDA-compatible driver/toolkit, Git, Python3.10 + Python3.12,
one explicitly allocated80/96GB GPU and adequate host RAM/disk.
GPU memory fit has not been newly benchmarked by this release.
For two complete native checkpoints plus an in-progress save, reserve roughly
150GiB free disk (more when reproducing SFT locally).

```bash
git clone https://github.com/leiye302/rethink-alfworld-qwen25-3b-no-local-hint
cd rethink-alfworld-qwen25-3b-no-local-hint
python3.10 -m venv .venv-sft
.venv-sft/bin/pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128
.venv-sft/bin/pip install -r configs/requirements-sft.txt
.venv-sft/bin/python src/tools/setup.py --framework --bundled-data

python3.12 -m venv .venv-opsd
.venv-opsd/bin/pip install --upgrade pip setuptools wheel packaging ninja
.venv-opsd/bin/pip install -r configs/requirements-opsd.txt
.venv-opsd/bin/pip install flash-attn==2.7.4.post1 --no-build-isolation
.venv-opsd/bin/pip install -e .deps/sdar -c configs/requirements-opsd.txt
.venv-opsd/bin/pip check
cp .env.example .env
set -a
source .env
set +a
.venv-opsd/bin/alfworld-download
```

Set `ALFWORLD_DATA` to the downloaded root containing `json_2.1.1` and
`logic`. Preparation verifies3553 native training games, disjoint from seen140,
and the game file hashes. No private server, author filesystem or credential
is required. See [detailed recipe](docs/idea.md).

## Exact common initialization

Read [Qwen Research License](configs/model_license.txt) first.
Weights are a modified Qwen2.5-3B model for non-commercial research/evaluation;
the source-code Apache license does not replace the model license.
The public release contains17 original model/tokenizer/manifest files
(about12.36GB), no optimizer states or private logs.

```bash
.venv-sft/bin/python src/tools/setup.py --download-shared-start --accept-model-license
# Or import an existing copy; every original file size/SHA is verified:
.venv-sft/bin/python src/tools/setup.py --shared-start /path/to/full-sft2000
```

The lock is `configs/shared_start.json`; the destination is
`checkpoints/route_sft/global_step_2000`. This **full-correction** checkpoint
is the common start for both Full and no-hint experiments.
Never substitute the trigger-only checkpoint, WebShop1500, a1.5B model or base.

Alternatively, the bundled full-correction SFT implementation/data can reproduce
a common initializer from the public pinned3B base. Both arms must then reuse
that **same resulting artifact**; separate same-recipe reruns are not guaranteed
identical. The default documented comparison uses the exact released initializer.

## Run only this ablation

```bash
bash scripts/reproduce.sh --check
.venv-sft/bin/python src/tools/audit_recipe.py --framework
.venv-opsd/bin/python src/tools/check.py --runtime opsd
bash scripts/reproduce.sh --run
# Same run only, after confirming old processes exited and checkpoint is complete:
bash scripts/reproduce.sh --run --resume
```

No API key or paid-call permission is requested. The shell and launch code reject
API/router enablement rather than silently turning it back on.
The controller requires the shared initialization and never auto-retrains SFT
as a fallback for missing/corrupt weights. SFT reproduction is explicit only.
It trains150 updates, exports update150, and evaluates fixed seen140.
It never launches another baseline. Keep the newest two complete RL checkpoints.

## Evaluation and comparison

```bash
EVAL_GPU=0 bash scripts/eval.sh --model checkpoints/opsd_hf150 \
  --out runs/independent_seen140 --step 150
.venv-sft/bin/python src/tools/plot.py --reference /path/to/full/output/summary.json \
  --ablation runs/independent_seen140/output/summary.json
```

Task list: `configs/seen140_tasks.json`, seeds2026..2165, history2, max50 actions,
temperature.4, sampling, n1, response512. API, Analyzer and Teacher are disabled
during evaluation. RETHINK rate is marked decisions / active decisions, not tokens.
The comparison accepts the two verified equivalent JSON serializations of the
same140 tasks; changed tasks/order/seeds/game hashes are still rejected.

CPU tests and clean-checkout checks are engineering validation, **not a new GPU
training result or success-rate claim**. No fabricated scores are shipped.
Inherited hint code remains an inactive reference; the active contract is
this README, `configs/ablation.json` and the frozen3B recipe.
