# Titulacion\_SigKAN\_Predictive

**Indoor Air Quality (IAQ) contaminant prediction using Sigmoidal Kolmogorov-Arnold Networks (SigKAN)**

Senior Thesis — Systems Engineering, Universidad de Lima
Authors: **Rodrigo Linares** · **Franco Gómez**

---

## Overview

This repository implements a **SigKAN Temporal** model for predicting four critical indoor air quality contaminants:

| Contaminant | Unit   | Relevance |
|---|---|---|
| CO₂         | ppm    | Ventilation / cognition |
| PM2.5       | µg/m³  | Respiratory health |
| PM10        | µg/m³  | Respiratory health |
| TVOC        | ppb    | Total volatile organic compounds |

The architecture combines a **TemporalMixer** (depthwise convolution + gated activation) with a **SigKAN** network (KAN with per-edge Sigmoid and SiLU activations) to capture complex temporal dependencies without the computational overhead of Transformers.

---

## Architecture

```
Input (B, W, F)
       │
 ┌─────▼──────┐     × num_mix_layers
 │TemporalMixer│  ← depthwise Conv1d + Sigmoid gate + LayerNorm
 └─────┬──────┘
       │ flatten → (B, W×F)
 ┌─────▼──────┐
 │   SigKAN   │  ← KAN layers with SigKANEdge (Sigmoid basis + SiLU residual)
 └─────┬──────┘
       │
  Prediction (B, 4)   # CO₂, PM2.5, PM10, TVOC
```

### Core components (`src/`)

| Module | Class | Role |
|---|---|---|
| `sigkan.py` | `SigKANEdge` | KAN edge: Sigmoid basis + SiLU residual |
| `sigkan.py` | `SigKANLayer` | Full KAN layer with LayerNorm |
| `sigkan.py` | `SigKAN` | Layer stack + dropout |
| `temporal_dataset.py` | `IAQTemporalPreprocessor` | Temporal 70/15/15 split, StandardScaler |
| `temporal_dataset.py` | `SlidingWindowDataset` | Sliding-window PyTorch Dataset |
| `sigkan_temporal.py` | `TemporalMixer` | Temporal mixing: depthwise conv + sigmoid gate |
| `sigkan_temporal.py` | `SigKANTemporal` | Full model + `.encode()` for latent extraction |

---

## Project structure

```
Titulacion_SigKAN_Predictive/
├── src/
│   ├── __init__.py
│   ├── sigkan.py                  # SigKANEdge, SigKANLayer, SigKAN
│   ├── temporal_dataset.py        # IAQTemporalPreprocessor, SlidingWindowDataset
│   └── sigkan_temporal.py         # TemporalMixer, SigKANTemporal
├── notebooks/
│   ├── co2/                       # CO₂ experiment
│   ├── pm2_5/                     # PM2.5 experiment
│   ├── pm10/                      # PM10 experiment
│   └── tvoc/                      # TVOC experiment
├── data/                          # Datasets (excluded from git)
├── models/                        # Trained checkpoints .pth (excluded from git)
├── results/figures/               # Output plots
├── scripts/
│   └── generate_notebooks.py      # Notebook generator
├── main.py                        # Quick smoke-test entrypoint
├── pyproject.toml                 # Dependencies managed with uv
└── uv.lock                        # Lock file for exact reproducibility
```

---

## Installation

Requires [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/solaremix/Titulacion_SigKAN_Predictive.git
cd Titulacion_SigKAN_Predictive
uv sync
```

> `uv sync` creates the `.venv` and installs the exact dependencies from `uv.lock`.

---

## Quick start

```python
from src import SigKAN, SigKANTemporal, IAQTemporalPreprocessor, SlidingWindowDataset
import torch

# Preprocessing
prep = IAQTemporalPreprocessor()
(X_train, y_train), (X_val, y_val), (X_test, y_test) = prep.fit_transform(df)

# Dataset with 24-step look-back window
train_ds = SlidingWindowDataset(X_train, y_train, window=24, horizon=1)

# Temporal model
model = SigKANTemporal(
    in_features=X_train.shape[1],
    window=24,
    hidden_sizes=[128, 64],
    out_features=4,        # CO₂, PM2.5, PM10, TVOC
    num_basis=8,
    num_mix_layers=2,
)

# Forward pass
x = torch.randn(32, 24, X_train.shape[1])
pred   = model(x)           # (32, 4)
latent = model.encode(x)    # (32, 64)

print(f"Parameters: {model.count_parameters():,}")
```

---

## Branch strategy

| Branch | Purpose |
|---|---|
| `main` | Production / stable releases |
| `develop` | Feature integration |
| `feature/setup-sigkan-core` | Initial setup, core architecture |
| `feature/english-translation` | Full codebase translated to English |

---

## Key dependencies

| Package | Version |
|---|---|
| torch | 2.11.0 |
| numpy | 2.4.4 |
| pandas | 3.0.2 |
| scikit-learn | 1.8.0 |
| matplotlib | 3.10.8 |
| ipykernel | 7.2.0 |

Managed with `uv` — exact reproducibility guaranteed via `uv.lock`.

---

## License

MIT — see [LICENSE](LICENSE) for details.
