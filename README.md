<h1 align="center">Data-Forcing Distillation (DFD) — Autoregressive Video Generation</h1>
<h3 align="center">DFD on top of Self-Forcing</h3>

<p align="center">
  Siyi Chen · Shaowei Liu · Yixuan Jia · Zian Wang · Huan Ling · Qing Qu · Jun Gao
</p>

<p align="center">
  <a href="https://csy2077.github.io/dfd-project-page/">Project Page</a> ·
  <a href="https://github.com/csy2077/data-forcing-distillation">Text-to-Video &amp; Image-to-Video code</a>
</p>

---

This repository holds the **autoregressive video generation** experiments for
**Data-Forcing Distillation (DFD)**, applied on top of the
[Self-Forcing](https://github.com/guandeh17/Self-Forcing) framework.

DFD is a simple post-training step that restores diversity and fidelity in
DMD-distilled few-step video generators. Standard DMD collapses sample diversity
and over-saturates outputs. DFD adds **explicit real-data supervision** to the
distribution-matching loss — evaluating the teacher score at a *real* sample
instead of the student's own generation — which amounts to a **single line of
code change**. Here we show the same idea transfers to the autoregressive,
streaming setting of Self-Forcing.

> This codebase is adapted from
> [Self-Forcing](https://github.com/guandeh17/Self-Forcing) (Huang et al., 2025).
> The text-to-video (Wan2.1) and image-to-video (Cosmos) experiments live in a
> separate repository:
> [data-forcing-distillation](https://github.com/csy2077/data-forcing-distillation).

## Installation

```bash
conda create -n self_forcing_dfd python=3.10 -y
conda activate self_forcing_dfd
pip install -r requirements.txt
pip install flash-attn --no-build-isolation
python setup.py develop
```

## Data and Checkpoints

- **Training data (ViPE long, LMDB):**
  [csusupergear/vipe_long_lmdb](https://huggingface.co/datasets/csusupergear/vipe_long_lmdb)
- **Pretrained DMD generator checkpoints (Self-Forcing):**
  [csusupergear/self_forcing_checkpoints](https://huggingface.co/datasets/csusupergear/self_forcing_checkpoints)

You will also need the Wan2.1-T2V-1.3B base model used by Self-Forcing:

```bash
huggingface-cli download Wan-AI/Wan2.1-T2V-1.3B \
  --local-dir-use-symlinks False --local-dir wan_models/Wan2.1-T2V-1.3B
```

## Configuration

DFD post-training runs from [`configs/self_forcing_dfd.yaml`](configs/self_forcing_dfd.yaml).
The key parameters to set:

```yaml
generator_ckpt: <path to the DMD checkpoint, e.g. .../model.pt>   # pretrained generator
resume_step: 1200                # training step of the pretrained generator_ckpt
data_path: <.../vipe_long_lmdb>  # path to the training video data
batch_size: 1
gradient_accumulation_steps: <N> # global batch = batch_size × num_gpus × grad_accum_steps
```

## Training

DFD post-training, starting from the pretrained DMD checkpoint:

```bash
torchrun --nproc_per_node=4 train.py \
  --config_path configs/self_forcing_dfd.yaml \
  --logdir logs/self_forcing_dfd_posttrain
```

## Inference

```bash
python inference.py \
  --config_path configs/self_forcing_dfd.yaml \
  --output_folder sf_videos/self_forcing_dfd_1360 \
  --checkpoint_path logs/self_forcing_dfd_posttrain/checkpoint_model_001360/model.pt \
  --data_path prompts/dfd_prompts.txt \
  --use_ema
```

The prompts used for our qualitative results are in
[`prompts/dfd_prompts.txt`](prompts/dfd_prompts.txt).

## Citation

If you find this useful, please cite both DFD and Self-Forcing:

```bibtex
@misc{dfd2026,
  title  = {Data-Forcing Distillation: Restoring Diversity and Fidelity
            in Few-Step Video Generation},
  author = {Chen, Siyi and Liu, Shaowei and Jia, Yixuan and Wang, Zian
            and Ling, Huan and Qu, Qing and Gao, Jun},
  year   = {2026}
}

@article{huang2025selfforcing,
  title   = {Self Forcing: Bridging the Train-Test Gap in Autoregressive Video Diffusion},
  author  = {Huang, Xun and Li, Zhengqi and He, Guande and Zhou, Mingyuan and Shechtman, Eli},
  journal = {arXiv preprint arXiv:2506.08009},
  year    = {2025}
}
```

## Acknowledgements

This codebase is built on top of
[Self-Forcing](https://github.com/guandeh17/Self-Forcing), which itself builds on
[CausVid](https://github.com/tianweiy/CausVid) and
[Wan2.1](https://github.com/Wan-Video/Wan2.1).
