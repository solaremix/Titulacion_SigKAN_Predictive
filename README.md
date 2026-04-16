# Titulación SigKAN + GNN

Proyecto de Titulación en Ingeniería de Sistemas — Universidad de Lima

**Autores:** Rodrigo Linares y Franco Gómez

## Descripción

Modelo SigKAN con redes neuronales de grafos (GNN) para predicción de calidad del aire interior (IAQ). Contaminantes: CO2, PM2.5, PM10, TVOC.

## Estructura

```
Titulacion_SigKAN_GNN/
├── data/               # Datasets (ignorados por git)
├── src/                # Módulos core
│   ├── sigkan.py
│   ├── temporal_dataset.py
│   └── sigkan_temporal.py
├── models/             # Checkpoints entrenados
├── results/
│   └── figures/        # Gráficas de resultados
├── notebooks/
│   ├── co2/
│   ├── pm2_5/
│   ├── pm10/
│   └── tvoc/
└── scripts/
    └── generar_notebooks.py
```

## Uso

```bash
pip install -r requirements.txt
python scripts/generar_notebooks.py
```
