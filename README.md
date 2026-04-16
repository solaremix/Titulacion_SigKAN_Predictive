# Titulacion\_SigKAN\_Predictive

**Predicción de contaminantes de calidad del aire interior (IAQ) mediante Sigmoidal Kolmogorov-Arnold Networks (SigKAN)**

Proyecto de Titulación — Ingeniería de Sistemas, Universidad de Lima  
Autores: **Rodrigo Linares** · **Franco Gómez**

---

## Descripción

Este repositorio implementa un modelo **SigKAN Temporal** para predecir cuatro contaminantes críticos de calidad del aire interior:

| Contaminante | Unidad  | Relevancia |
|---|---|---|
| CO₂          | ppm     | Ventilación / cognición |
| PM2.5        | µg/m³   | Salud respiratoria |
| PM10         | µg/m³   | Salud respiratoria |
| TVOC         | ppb     | Compuestos orgánicos volátiles |

La arquitectura combina un **TemporalMixer** (convolución depthwise + gated activation) con una red **SigKAN** (KAN con activaciones Sigmoid y SiLU por arista) para capturar dependencias temporales complejas sin el costo computacional de Transformers.

---

## Arquitectura

```
Entrada (B, W, F)
       │
 ┌─────▼──────┐     x num_mix_layers
 │TemporalMixer│  ← depthwise Conv1d + Sigmoid gate + LayerNorm
 └─────┬──────┘
       │ flatten → (B, W×F)
 ┌─────▼──────┐
 │  SigKAN    │  ← capas KAN con SigKANEdge (Sigmoid basis + SiLU residual)
 └─────┬──────┘
       │
   Predicción (B, 4)   # CO₂, PM2.5, PM10, TVOC
```

### Componentes core (`src/`)

| Módulo | Clase | Rol |
|---|---|---|
| `sigkan.py` | `SigKANEdge` | Arista KAN: basis Sigmoid + residual SiLU |
| `sigkan.py` | `SigKANLayer` | Capa KAN completa con LayerNorm |
| `sigkan.py` | `SigKAN` | Stack de capas + dropout |
| `temporal_dataset.py` | `IAQTemporalPreprocessor` | Split 70/15/15 temporal, StandardScaler |
| `temporal_dataset.py` | `SlidingWindowDataset` | Ventanas deslizantes para PyTorch |
| `sigkan_temporal.py` | `TemporalMixer` | Mixing temporal con conv depthwise + gate |
| `sigkan_temporal.py` | `SigKANTemporal` | Modelo completo + método `.encode()` |

---

## Estructura del proyecto

```
Titulacion_SigKAN_Predictive/
├── src/
│   ├── __init__.py
│   ├── sigkan.py              # SigKANEdge, SigKANLayer, SigKAN
│   ├── temporal_dataset.py    # IAQTemporalPreprocessor, SlidingWindowDataset
│   └── sigkan_temporal.py     # TemporalMixer, SigKANTemporal
├── notebooks/
│   ├── co2/                   # Experimento CO₂
│   ├── pm2_5/                 # Experimento PM2.5
│   ├── pm10/                  # Experimento PM10
│   └── tvoc/                  # Experimento TVOC
├── data/                      # Datasets (excluido de git)
├── models/                    # Checkpoints .pth (excluido de git)
├── results/figures/           # Gráficas de resultados
├── scripts/
│   └── generar_notebooks.py
├── main.py                    # Entrypoint de validación rápida
├── pyproject.toml             # Dependencias gestionadas con uv
└── uv.lock                    # Lock file para reproducibilidad exacta
```

---

## Instalación

Requiere [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/solaremix/Titulacion_SigKAN_Predictive.git
cd Titulacion_SigKAN_Predictive
uv sync
```

> `uv sync` crea el `.venv` e instala las dependencias exactas del `uv.lock`.

---

## Uso rápido

```python
from src import SigKAN, SigKANTemporal, IAQTemporalPreprocessor, SlidingWindowDataset
import torch

# Preprocesamiento
prep = IAQTemporalPreprocessor()
(X_train, y_train), (X_val, y_val), (X_test, y_test) = prep.fit_transform(df)

# Dataset con ventana de 24 pasos
train_ds = SlidingWindowDataset(X_train, y_train, window=24, horizon=1)

# Modelo temporal
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
pred = model(x)            # (32, 4)
latent = model.encode(x)   # (32, 64)

print(f"Parámetros: {model.count_parameters():,}")
```

---

## Ramas

| Rama | Propósito |
|---|---|
| `main` | Producción / releases estables |
| `develop` | Integración de features |
| `feature/setup-sigkan-core` | Setup inicial, arquitectura core |

---

## Dependencias principales

| Paquete | Versión |
|---|---|
| torch | 2.11.0 |
| numpy | 2.4.4 |
| pandas | 3.0.2 |
| scikit-learn | 1.8.0 |
| matplotlib | 3.10.8 |
| ipykernel | 7.2.0 |

Gestionadas con `uv` — reproducibilidad garantizada vía `uv.lock`.

---

## Licencia

MIT — ver [LICENSE](LICENSE) para detalles.
