# RAM：面向文本到动作生成的重建锚定扩散模型

[English](README.md) | **中文**

**RAM（Reconstruction-Anchored Diffusion Model）** 的官方 PyTorch 实现，用于文本到动作（text-to-motion）生成。

- 📄 论文：[arXiv:2601.14788](https://arxiv.org/pdf/2601.14788)
- 🌐 项目主页：https://feifeifeiliu.github.io/RAM/

RAM 在文本到动作的扩散模型上额外协同训练一个**动作重建分支**，并通过两个目标塑造运动中心的潜在空间——**自正则化**（`L_sr`）与**运动中心潜在对齐**（`L_latent`）；推理阶段用 **Reconstructive Error Guidance (REG)** 借助重建分支，引导采样远离易错估计。本仓库提供 **HumanML3D** 上文本到动作的训练与评估。

---

## 1. 环境

Python 3.8。用 `environment.yml` 创建 conda 环境（环境名 `ram`），或在自有 Python 3.8 环境中用 `requirement.txt` 安装：

```bash
conda env create -f environment.yml
conda activate ram
# 或安装到已有环境：
# pip install -r requirement.txt
```

> **注意**：`eval/eval_humanml_cldiff.py` 顶部把 `HF_ENDPOINT` / `HF_HUB_URL` 指向了 Hugging Face 镜像、`WANDB_BASE_URL` 指向了代理。如可直连，请自行修改或删除这几行。

## 2. 数据与依赖资源

RAM 复用 **Motion Diffusion Model (MDM)** 的 HumanML3D 数据管线与 T2M 评估器。请按 MDM 原仓库的数据准备说明，在本仓库根目录下准备好以下目录：

- `dataset/` —— HumanML3D（以及可选的 KIT-ML）动作数据
- `glove/` —— T2M 评估器使用的 GloVe 词向量
- `body_models/` —— SMPL 人体模型
- `t2m/` —— 预训练的 T2M 评估模型（用于计算指标）

MDM 原仓库（数据与评估器准备）：https://github.com/GuyTevet/motion-diffusion-model

请把各资源放到上述对应路径。

## 3. 预训练权重

权重托管在 ModelScope：**[liuyifei96/RAM](https://modelscope.cn/models/liuyifei96/RAM)**。

下载到 `save/t2m/RAM/`：

```bash
pip install modelscope
modelscope download --model liuyifei96/RAM --local_dir save/t2m/RAM
```

得到的目录结构：

```
save/t2m/RAM/
├── model000450000.pt
├── my_setting.yaml
└── args.json
```

`my_setting.yaml` 保存训练配置，评估时会自动从该 checkpoint 所在目录读取。

## 4. 训练

```bash
OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=0 python -m train.mdm_tmr.train_mdm \
  --save_dir save/t2m/my_run \
  --dataset humanml \
  --diffusion_steps 50 \
  --noise_schedule cosine \
  --mask_frames \
  --use_ema \
  --num_steps 500000 \
  --weight_decay 1e-6 1e-3 \
  --train_platform_type NoPlatform \
  --overwrite
```

- 默认开启混合精度（AMP）与 EMA；`--weight_decay` 接收两个值（decoder / encoders）。
- 创建 `--save_dir` 时会把当前配置 `configs/my_setting.yaml` 复制进去；每 5 万步保存一次 checkpoint。
- 需要用 Weights & Biases 记录实验时，使用 `--train_platform_type WandBPlatform --project_name <name>`。

## 5. 评估

```bash
OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=0 python -m eval.eval_humanml_cldiff \
  --model_path save/t2m/RAM/model000450000.pt \
  --guidance_param 1.5 5.0 \
  --diffuser_name ddpm \
  --inference_step 20 \
  --dynamic_cfg False \
  --eval_mode mm_short
```

- `--guidance_param w_cfg w_reg`：两个标量——第一个是 **CFG** 权重（`w2`），第二个是 **REG** 权重（`w1`）。论文设置为 `1.5 5.0`。
- `--eval_mode`：`debug`（1k 样本，快速）、`wo_mm`（5k 样本 × 20 次重复，不含 MultiModality）、`mm_short`（5k × 20 次重复 + MultiModality，对应论文 Table 1 全部列）、`aits`（推理耗时基准）。
- `--inference_step`：去噪步数（论文为 20）；`--diffuser_name` 选择采样器（如 `ddpm`）。

## 6. 配置

`configs/my_setting.yaml` 定义训练配置，关键项如下：

| 字段 | 含义 |
|---|---|
| `train_config.beta` | 运动中心潜在对齐（`L_latent`）的梯度调制系数，默认 `0.01` |
| `train_config.loss_weight.sr` | 自正则化 `L_sr` 的权重，默认 `1.0` |
| `train_config.loss_weight.latent` | `L_latent` 的权重，默认 `0.5` |
| `train_config.loss_weight.sub_weight.{gen,rec}` | 生成/重建数据损失的子权重 |
| `train_config.temperature`、`threshold_selfsim` | 自正则化温度与近义句过滤阈值 |
| `diffusion_config.diffusion_steps` | 扩散步数（50） |
| `model_config.{decoder,text_encoder,motion_encoder,ram}` | 各子模块配置（扩散解码器、`E_t`、`E_m` 及 RAM 封装） |

## 7. 仓库结构

```
model/         # RAM、扩散解码器（MDMDecoder）、编码器
diffusion/     # 高斯扩散、diffusers 采样管线、REG+CFG 引导
train/         # 训练循环与入口
eval/          # HumanML3D 评估
data_loaders/  # 数据集、文本处理、动作加载器
utils/         # 配置、模型构建、采样器、杂项工具
configs/       # my_setting.yaml 与采样器参数
```

## 8. 致谢与引用

本代码基于并借鉴了以下项目：

- **MDM —— Motion Diffusion Model**：https://github.com/GuyTevet/motion-diffusion-model
- **TMR —— Text-to-Motion Retrieval**：https://github.com/Mathux/TMR
- **TEMOS**：https://github.com/Mathux/TEMOS

感谢上述作者开源代码。

如果 RAM 对你有帮助，请引用：

```bibtex
@inproceedings{liu2026ram,
  title        = {Reconstruction-Anchored Diffusion Model for Text-to-Motion Generation},
  author       = {Liu, Yifei and Ding, Changxing and Guo, Ling and Jiang, Huaiguang and Cao, Qiong},
  booktitle    = {European Conference on Computer Vision (ECCV)},
  pages        = {608--625},
  year         = {2026},
  organization = {Springer}
}
```

## 9. 许可

本项目以 MIT 许可发布（见 [`LICENSE`](LICENSE)）；源自 MDM / TMR / TEMOS 的部分仍受其各自许可约束。
