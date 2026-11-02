# Engineering correction

The mistakenly WebShop-targeted no-hint control has been replaced by
ALFWorld Qwen2.5-3B, shared full Route-SFT2000 initialization.
The original Git history is retained; the active tree/configuration is ALFWorld.

- Stage2 local_hints.enabled=false; native static-skill Teacher remains.
- No API key, router, Analyzer or paid hint request in the active launch path.
- Full SFT weights/tokenizer are pinned per file; no base/trigger/WebShop substitution.
- Shared native GRPO/OPSD loss files are hash-checked against the public reference.
- Python PATH/relative-path lookup and equivalent seen140 serialization are fixed.
- Regression tests reject changed evaluation identity and incorrect initialization.

Preparation runs CPU tests, clean exported-source checks, and privacy/history
scans. These are not GPU training, memory-fit or performance claims.
Real GPU preflight/150-update execution requires a separately allocated device.
