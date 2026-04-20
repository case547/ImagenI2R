# ImagenI2R Codebase Overview

Generates **regular time series from irregularly-sampled data** using a two-stage approach:
1. A **Time Series Transformer (TST)** completes/imputes irregular sequences
2. A **vision-based diffusion model** (EDM/Karras formulation) generates realistic time series by treating them as images

## Architecture

| Component | File | Role |
|---|---|---|
| Main diffusion model | `models/our.py` — `TS2img_Karras` | EDM-style denoising with masked noise for irregular data |
| U-Net backbone | `models/networks.py` — `EDMPrecond` | Image-space denoising network |
| TS↔Image conversion | `models/img_transformations.py` — `DelayEmbedder` | Encodes time series as delay-embedded images |
| TST encoder/decoder | `models/TST.py`, `models/decoder.py` | Transformer for sequence completion |
| Sampler | `models/sampler.py` — `DiffusionProcess` | EDM stochastic/deterministic sampling |
| EMA | `models/ema.py` | Exponential moving average for stable training |

## Entry Point & Config

- `run_irregular.py` — training/evaluation loop
- `configs/seq_len_{24,96,768}/[dataset].yaml` — per-dataset YAML configs
- Datasets: Stock, ETTh1/h2, ETTm1/m2, Weather, Energy, Electricity, MuJoCo, Sine

## Metrics (`metrics/`)

- Context-FID, discriminative score, predictive score, correlation score

## Utils

- `utils/utils_data.py` — data loading
- `utils/utils_args.py` — argument parsing
- `utils/loggers/` — pluggable logging (Neptune, MLflow, TensorBoard, print)
- `utils/persistence.py` — checkpoint save/restore
