# Uncertainty-aware Blur Prior (UBP)

Official code for [Bridging the Vision-Brain Gap with an Uncertainty-Aware Blur Prior](https://arxiv.org/abs/2503.04207) (CVPR 2025), plus the manuscript Results pipeline (THINGS-EEG / Alljoined).

## Directory roles (read this first)

One job per place — avoid treating similar-looking folders as duplicates:

| Path | Role |
|------|------|
| `data/` | **数据**：`things-eeg/` 文件 + `registry.yaml` / `profiles/` 注册表 |
| `preprocess/` + `scripts/*preprocess*` | 把原始数据转成训练用格式 |
| `encoder/` + `configs/eeg/` | EEG **编码器**：`python -m encoder.train`（或 `scripts/train_things_encoder.sh` / `train_alljoined.sh`） |
| `predictor/` | EEG **预测器**（图像→EEG）：`python -m predictor.train`（或 `scripts/train_*two_stage*.sh`） |
| `checkpoints/encoder/` | **编码器权重**（`THINGSEEG2/` / `Alljoined/`） |
| `checkpoints/predictor/` | **预测器（生成器）权重**（`Ours` + 消融） |
| `pretrained/` | CLIP 等第三方预训练权重 |
| `analysis/eeg_gen_eval/` | 论文评估：`plots/` / `helpers/` / `compute/` + 缓存 `raw*` + 成图 `figures/` |
| `manuscript/` | 手稿 TeX（引用 `analysis/.../figures`） |
| `scripts/` | 一键入口（优先用这里，不要散落找命令） |

## Environment

```bash
pip install -r requirements.txt
# scripts 默认 conda 环境名: UBP
```

## Data

统一入口：`data/`（文件 + `registry.yaml` / `profiles/`）。环境变量见 `data/env.example`。

```bash
python -m encoder.registry list
python -m encoder.registry show things_eeg
```

**THINGS-EEG**（仓库内 `data/things-eeg/`，或 HF：[Haitao999/things-eeg](https://huggingface.co/datasets/Haitao999/things-eeg)）

若只有原始数据，再跑预处理（已下载 `Preprocessed_data_250Hz_whiten/` 可跳过）：

```bash
python preprocess/process_resize.py --type eeg
for s in $(seq 1 10); do
  python preprocess/process_eeg_whiten.py --subject "$s"
done
```

**Alljoined-1.6M**（HF：[Alljoined/Alljoined-1.6M](https://huggingface.co/datasets/Alljoined/Alljoined-1.6M) → UBP `.pt`）

| Item | Value |
|------|-------|
| Registry id | `alljoined_eeg` |
| Shape | train `(16540, 4, 32, 250)` / test `(200, 80, 32, 250)` @ 250 Hz |
| Images | 复用 `data/things-eeg/Image_set_Resize` |
| Channels | `encoder.data.ALLJOINED_CHANNELS`（32） |

```text
$UBP_EEG_DATA_ROOT/alljoined-1.6M/          # 默认 /home/ubuntu/dataset/EEG
├── raw_hf/preprocessed_eeg/               # 下载缓存（可删，仅预处理需要）
└── ubp_preprocessed/sub-XX/{train,test}.pt
```

```bash
export UBP_EEG_DATA_ROOT=/home/ubuntu/dataset/EEG   # 见 data/env.example
# optional: export HF_ENDPOINT=https://hf-mirror.com
bash scripts/download_alljoined.sh
bash scripts/preprocess_alljoined_all.sh
python -m encoder.registry show alljoined_eeg
```

Notes: 官方 MVNN 白化不必与 THINGS `process_eeg_whiten.py` 数值一致；测试集每图取前 80 trial。

## Reproduce manuscript Results

```bash
# THINGS: encoder → generator
bash scripts/train_things_encoder.sh
bash scripts/train_final_two_stage.sh

# Fig.4 ablations（训练；也可 run_*_ablation.sh = 训练 + 聚合 + 重画 final_fig4）
bash scripts/train_structural_ablation.sh   # no_Dilated, no_Transformer
bash scripts/train_extended_ablation.sh     # no_FoveaBlur, no_self_attn, h128, h512
bash scripts/train_loss_group_ablation.sh

# Alljoined: encoder (20) → generator (strong-5)
bash scripts/train_alljoined.sh
bash scripts/train_alljoined_final_two_stage.sh

# Redraw paper figures (uses analysis/eeg_gen_eval/raw*)
PYTHONPATH=. python -m analysis.eeg_gen_eval.plots.redraw_all

# Compile manuscript excerpt
cd manuscript && xelatex results_sec3.tex
```

## Citation

```bibtex
@inproceedings{wu2025bridging,
  title={Bridging the Vision-Brain Gap with an Uncertainty-Aware Blur Prior},
  author={Wu, Haitao and Li, Qing and Zhang, Changqing and He, Zhen and Ying, Xiaomin},
  booktitle={Proceedings of the Computer Vision and Pattern Recognition Conference},
  pages={2246--2257},
  year={2025}
}

@misc{xu2025alljoined16mmilliontrialeegimagedataset,
  title={Alljoined-1.6M: A Million-Trial EEG-Image Dataset for Evaluating Affordable Brain-Computer Interfaces},
  author={Jonathan Xu and others},
  year={2025},
  eprint={2508.18571},
}
```

## Contact

yangjiaqi_bme2026@163.com
