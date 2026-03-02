# CED GPU Pack v12 (full dataset default)

## What changed vs v11
- Fixed auto dataset discovery selecting random json under conda env/site-packages.
- gpu.sh is aligned with cpu.sh (same HF_HOME defaults, same model resolve step).
- Default dataset discovery only searches:
  - ../dataprepare/data
  - ../dataprepare/datasets
  and run_full_dataset.py will also exclude conda_envs/site-packages and validate candidates by prompt_ratio.

## Usage
- Default: auto select dataset under dataprepare/data|datasets
  bash gpu.sh

- Explicit dataset
  bash gpu.sh --data /abs/path/to/dataset.jsonl

- Smoke test
  bash gpu.sh --smoke

Outputs:
- logs/dataset_run_*/predictions.jsonl
- logs/dataset_run_*/summary.json
