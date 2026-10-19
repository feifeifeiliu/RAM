# RAM: Reconstruction-Anchored Diffusion Model for Text-to-Motion Generation

**English** | [中文](README_zh.md)

Official PyTorch implementation of **RAM (Reconstruction-Anchored Diffusion Model)** for text-to-motion generation.

- 📄 Paper: [arXiv:2601.14788](https://arxiv.org/pdf/2601.14788)
- 🌐 Project page: https://feifeifeiliu.github.io/RAM/

RAM augments a text-to-motion diffusion model with a co-trained **motion reconstruction branch**. Two objectives shape a motion-centric latent space — **self-regularization** (`L_sr`) and **motion-centric latent alignment** (`L_latent`) — and at inference, **Reconstructive Error Guidance (REG)** uses the reconstruction branch to steer sampling away from error-prone estimates. This repository covers training and evaluation on **HumanML3D** (text-to-motion).

---

## 1. Environment

Python 3.8. Create the conda environment from `environment.yml` (env name: `ram`), or install into your own Python 3.8 environment with `requirement.txt`:

```bash
conda env create -f environment.yml
conda activate ram
# or, into an existing env:
# pip install -r requirement.txt
```

> **Note**: `eval/eval_humanml_cldiff.py` sets `HF_ENDPOINT` / `HF_HUB_URL` to a Hugging Face mirror and `WANDB_BASE_URL` to a proxy. Edit or remove these lines if you have direct access.

## 2. Data & Pretrained Assets

RAM reuses the standard HumanML3D data pipeline and the T2M evaluators from **Motion Diffusion Model (MDM)**. Prepare the following under the repository root by following the data-setup instructions in the MDM repository:

- `dataset/` — HumanML3D (and optionally KIT-ML) motion data
- `glove/` — GloVe embeddings used by the T2M evaluator
- `body_models/` — SMPL body models
- `t2m/` — pretrained T2M evaluation models (for metrics)

MDM repository (data & evaluator setup): https://github.com/GuyTevet/motion-diffusion-model

Place each resource at the path above.

## 3. Pretrained Checkpoint

The checkpoint is hosted on ModelScope: **[liuyifei96/RAM](https://modelscope.cn/models/liuyifei96/RAM)**.

Download it into `save/t2m/RAM/`:

```bash
pip install modelscope
modelscope download --model liuyifei96/RAM --local_dir save/t2m/RAM
```

Resulting layout:

```
save/t2m/RAM/
├── model000450000.pt
├── my_setting.yaml
└── args.json
```

`my_setting.yaml` holds the training configuration and is read automatically during evaluation (from the checkpoint's directory).

## 4. Training

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

- Training uses mixed precision (AMP) and EMA by default; `--weight_decay` takes two values (decoder / encoders).
- The active config `configs/my_setting.yaml` is copied into `--save_dir` when it is created, and checkpoints are saved every 50k steps.
- For experiment tracking with Weights & Biases, use `--train_platform_type WandBPlatform --project_name <name>`.

## 5. Evaluation

```bash
OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=0 python -m eval.eval_humanml_cldiff \
  --model_path save/t2m/RAM/model000450000.pt \
  --guidance_param 1.5 5.0 \
  --diffuser_name ddpm \
  --inference_step 20 \
  --dynamic_cfg False \
  --eval_mode mm_short
```

- `--guidance_param w_cfg w_reg`: two scalars — the first is the **CFG** weight (`w2`), the second is the **REG** weight (`w1`). The paper setting is `1.5 5.0`.
- `--eval_mode`: `debug` (1k samples, quick), `wo_mm` (5k samples × 20 reps, no MultiModality), `mm_short` (5k × 20 reps + MultiModality, full Table 1), `aits` (inference-time benchmark).
- `--inference_step`: number of denoising steps (paper uses 20); `--diffuser_name` selects the sampler (e.g. `ddpm`).

## 6. Configuration

`configs/my_setting.yaml` defines the training setup. Key fields:

| Field | Meaning |
|---|---|
| `train_config.beta` | gradient modulator for motion-centric latent alignment (`L_latent`), default `0.01` |
| `train_config.loss_weight.sr` | weight of self-regularization `L_sr`, default `1.0` |
| `train_config.loss_weight.latent` | weight of `L_latent`, default `0.5` |
| `train_config.loss_weight.sub_weight.{gen,rec}` | sub-weights of the generation / reconstruction data losses |
| `train_config.temperature`, `threshold_selfsim` | self-regularization temperature and near-duplicate filtering threshold |
| `diffusion_config.diffusion_steps` | number of diffusion steps (50) |
| `model_config.{decoder,text_encoder,motion_encoder,ram}` | sub-module configs (diffusion decoder, `E_t`, `E_m`, and the RAM wrapper) |

## 7. Repository Structure

```
model/         # RAM, diffusion decoder (MDMDecoder), encoders
diffusion/     # Gaussian diffusion, diffusers sampling pipeline, REG+CFG guidance
train/         # training loop and entry point
eval/          # HumanML3D evaluation
data_loaders/  # datasets, text processing, motion loaders
utils/         # config, model builder, samplers, misc helpers
configs/       # my_setting.yaml and sampler params
```

## 8. Acknowledgements & Citation

This codebase builds upon and borrows from:

- **MDM — Motion Diffusion Model**: https://github.com/GuyTevet/motion-diffusion-model
- **TMR — Text-to-Motion Retrieval**: https://github.com/Mathux/TMR
- **TEMOS**: https://github.com/Mathux/TEMOS

We thank the authors for releasing their code.

If you find RAM useful, please cite:

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

## 9. License

This project is released under the MIT License (see [`LICENSE`](LICENSE)). Portions derived from MDM, TMR, and TEMOS remain under their respective licenses.
