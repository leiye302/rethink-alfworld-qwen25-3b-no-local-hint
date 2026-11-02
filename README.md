# WebShop3B ablation: same Route-SFT start, ordinary GRPO+OPSD

Same-start, **no-local-hint** control for [RETHINK WebShop3B](https://github.com/leiye302/rethink-webshop-qwen25-3b),
matched to commit `cb6e5d93929c4e33bb431b0cdcf353ea42196a91`.
Both arms start from the **exact same public Route-SFT1500 checkpoint**.
This is not a base-start baseline, pure GRPO or a fresh SFT run.

| Component | Main method | This ablation |
|---|---|---|
| Actor start | Released3B Route-SFT1500 | Same files/SHA |
| GRPO + ordinary OPSD / static skills | Enabled | Unchanged |
| Teacher | Current pre-update Actor, no-gradient scoring | Unchanged |
| Extra local hint at Actor RETHINK | Enabled | **Disabled; no API/Analyzer** |
| Registered RETHINK | Normal free generation | Retained, neither forced nor suppressed |
| Beta / OPSD coefficient / reference KL | 5 / .01 / .01 | Unchanged |
| Training | 150 updates,16 tasks ×8 rollouts | Unchanged |
| Validation | Initial/every10/final, shared140 | Same goals, seeds, sampling |
| Saves | Every20/final150, newest2 | Unchanged |
| Hardware | 4×A10080GB; FSDP4, TP2×DP2 | Unchanged |

`configs/ablation.json` freezes the complete matched recipe and55 shared files.
The only algorithm-configuration difference is
`algorithm.local_hints.enabled: true -> false`.
Loss/reward/prompt/environment, rollout budgets, tokenizer and evaluation are unchanged.

## 1. Standalone installation (Linux)

Requires an allocated4×A10080GB node, CUDA12.8-compatible driver, CUDA toolkit
for FlashAttention, Conda and Git. Do not use another job's GPUs.
No author computer, private server, data path or API credential is required.

```bash
git clone https://github.com/leiye302/rethink-webshop-qwen25-3b-no-local-hint
cd rethink-webshop-qwen25-3b-no-local-hint
conda create -y -p "$PWD/.venv-opsd" -c conda-forge python=3.10 openjdk=11
conda activate "$PWD/.venv-opsd"
python -m pip install --upgrade pip setuptools wheel packaging==25.0
python scripts/install_opsd.py
python -m pip install --no-build-isolation flash-attn==2.8.3
export PYTHONPATH="$PWD/src:$PWD/.deps/sdar"
python -m rethink_webshop.java --prepare
python -m rethink_webshop.setup --framework --data
python -m pip install --no-deps -e .deps/sdar
python -m pip check
export JAVA_HOME="$PWD/assets/java"
export JVM_PATH="$JAVA_HOME/lib/server/libjvm.so"
bash scripts/prepare.sh
python -m rethink_webshop.ablation
python -m pytest -q
python scripts/smoke_native.py
```

The public SDAR source pin remains `d511f043afe9199b8576fc99ea52e12d84841f32`.
Native loss/Teacher/prompt/parser/reward/static-skill hashes and the same thin
shared140/telemetry patch are unchanged. Preparation does not start training.

## 2. Exact shared initialization

Read and accept the [Qwen Research License](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct/blob/aa8e72537993ba99e69dfaafa59ed015b17504d1/LICENSE)
before downloading; these3B model weights are not Apache-2.0.

```bash
python -m rethink_webshop.setup --checkpoint1500 --accept-model-license
```

`configs/checkpoint1500.json` downloads the same21 files from the declared public
[main-method Route-SFT1500 release](https://github.com/leiye302/rethink-webshop-qwen25-3b/releases/tag/route-sft1500-v1),
checking every size/SHA. This includes original7 FP32 weight shards (about12.34GB),
tokenizer and manifest. No second repository needs to be cloned or installed.
Do not retrain SFT: the same recipe does not guarantee identical initial weights.
Any explicit `--actor` must contain the same checksum-pinned artifact.

## 3. Train150 updates, no API setup

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/train_opsd.sh
# Resume only the same complete ablation run.
CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/train_opsd.sh --resume
```

Default Actor: `assets/route-sft1500`. Output:
`runs/sft_start_grpo_opsd/{checkpoints,rollouts,validation}`.
All decisions use the ordinary native `_compute_teacher_log_probs` branch,
including decisions where the Actor generates RETHINK. Static skills remain
Teacher-only; Student context/responses/rewards and native losses are unchanged.
Teacher uses current Actor weights, not a frozen initial model.

**Do not use `--baseline`**: in the main repository it means base-start and is
rejected here. There is no `--allow-api-cost` training flag or key loading.
Inherited hint modules/tests remain reference code, not an active training path.
Do not mix main-method/base-start run directories or bypass resume identity checks.
Reserve disk for the initialization, dependencies/assets, two complete native
checkpoints and an in-progress save. Latest2 retention is unchanged.

## 4. Same fixed140 evaluation

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/eval.sh \
  --model assets/route-sft1500 --output runs/eval_initial
# Evaluate trained complete HF exports with the same entry point.
```

Uses the same140 synthetic native WebShop goals, explicitly excluded from
training, and worker seeds1000..1139. It is not the official human500.
Temperature.4, sampling, top-p1, unrestricted top-k, n1, history2, prompt4096,
response512 and15 actions are unchanged. Evaluation has no API or forced route.
See [shared140 contract](docs/SHARED_EVALUATION140.md). Report success and task
score separately; marker rate is a fraction of active decisions, not tokens.

## Contract, interpretation and verification

- [Exact ablation and invariants](docs/ABLATION.md)
- `configs/ablation.json`: immutable parent, matched recipe and shared source/data locks.
- `configs/checkpoint1500.json`: exact public initialization.
- [Third-party licenses/resources](THIRD_PARTY.md).

CPU tests exercise the exact native no-hint Teacher branch, same-start guards,
no-API preparation and inherited contracts. Clean CI also checks native WebShop
reset/search/feedback. This is code/configuration publication, not a new GPU
training result, memory-fit claim or performance score. Inherited main-method
documents are labeled references; this README and ABLATION.md define the active control.
