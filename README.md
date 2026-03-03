# CED P0 验证实验

## 目标
验证 CED (Counterfactual Entropy Divergence) 信号能否在特征空间区分
"模型真正使用了视觉信息" vs "依赖文本先验猜测"。

## CED 核心公式
```
CED(x, R) = JS(P_orig || P_replaced) + λ_e · max(0, H(P_replaced) - H(P_orig))
```
- 对图像中区域 R 对应的 visual token 替换为周围 token 均值
- JS: 替换前后输出分布的 Jensen-Shannon 散度
- 熵正则项: 替换后熵增 → 模型确实依赖了被替换区域

## 目录布局（与原有 hhj-train 结构对齐）
```
.../hhj-train/
├── dataprepare/                           # 共享硬盘（CPU/GPU都能访问）
│   ├── conda_envs/ced_p0/                 # Conda 环境
│   ├── models/Qwen3-VL-8B-Instruct/       # 模型权重
│   ├── data/
│   │   ├── coco/                          # COCO val2017 图片+标注
│   │   │   ├── val2017/
│   │   │   └── annotations/
│   │   └── ced_vqa/                       # 生成的四类VQA数据集
│   │       └── ced_vqa_dataset.jsonl
│   └── pip_wheels/                        # pip 离线 wheel 缓存
│
└── ced_p0/                                # ← 本项目代码放这里
    ├── cpu_download.sh
    ├── gpu_run.sh
    ├── requirements.txt
    ├── src/
    │   ├── model_loader.py
    │   ├── coco_vqa_gen.py
    │   ├── visual_token_map.py
    │   ├── ced_core.py
    │   ├── p0a_probe.py
    │   ├── p0b_validate.py
    │   └── analysis.py
    ├── results/                           # 实验输出（gpu_run.sh 自动创建）
    └── logs/                              # 日志
```

## 使用

### 1. CPU服务器（有网）
```bash
cd .../hhj-train/ced_p0
bash cpu_download.sh
```
自动下载：模型 + COCO val2017 + pip wheels + 生成VQA数据集

### 2. GPU服务器（无网，8×H200）
```bash
cd .../hhj-train/ced_p0
bash gpu_run.sh               # 完整流程：P0-a → P0-b → 分析
bash gpu_run.sh --probe-only  # 仅P0-a架构探测（快速验证，几分钟）
bash gpu_run.sh --skip-probe  # 跳过探测直接跑P0-b
bash gpu_run.sh --analysis    # 仅对已有结果跑分析
bash gpu_run.sh --smoke       # 只验活（与原gpu.sh兼容）
```

## 输出
- `results/p0a_probe_report.json` — 架构探测（hook/token映射是否可用）
- `results/p0b_raw.jsonl`         — 逐样本CED计算结果
- `results/p0b_summary.json`      — 汇总统计
- `results/p0b_analysis_report.json` — AUC-ROC + 最优公式/层
- `results/figures/`              — 可视化图表

## 关键路径环境变量（可覆盖默认值）
```bash
CED_ENV=...    # Conda 环境路径
LOCAL_DIR=...  # 模型目录
DEVICE=...     # GPU 设备（默认 cuda:0）
DTYPE=...      # 精度（默认 bfloat16）
```
