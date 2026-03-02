# CED GPU Pack v11

## 默认行为
`bash gpu.sh` 默认会：
1) 解析并确认本地模型目录（不下载，不触网）
2) 自动找到一个数据集文件（或使用你指定的 `--data` / `DATA_PATH`）
3) 跑完整数据集，输出结果到 `./logs/dataset_run_<ts>/`

输出：
- `predictions.jsonl`：每条样本一行（含 pred / gt / exact_match / js_first_token 等）
- `summary.json`：整体统计

## 常用命令
### 1) 默认跑整数据集
```bash
bash gpu.sh
```

### 2) 指定数据集路径
```bash
bash gpu.sh --data /abs/path/to/test.jsonl
# 或
DATA_PATH=/abs/path/to/test.jsonl bash gpu.sh
```

### 3) 只验活（不跑数据集）
```bash
bash gpu.sh --smoke
```

### 4) 跑你自己的脚本（自动替换 {MODEL_PATH}）
```bash
bash gpu.sh python your_exp.py --model_path {MODEL_PATH} --device cuda:0
```

## 说明
- `run_full_dataset.py` 是一个“通用评测壳子”，尽量兼容多种 jsonl/json schema。
- 如果你的数据集字段名很特殊，把 `pick_prompt / load_image_any / pick_gt` 这三段按你的字段名改一下就行。
