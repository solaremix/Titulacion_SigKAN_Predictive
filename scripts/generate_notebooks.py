"""
Experiment notebook generator for the SigKAN Predictive project.
Authors: Rodrigo Linares and Franco Gomez — Universidad de Lima
"""

import nbformat
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

CONTAMINANTS = [
    {
        "key": "co2",
        "label": "CO2 (Carbon Dioxide)",
        "folder": "notebooks/co2",
        "filename": "experiment_co2.ipynb",
        "target_col": "co2",
    },
    {
        "key": "pm2_5",
        "label": "PM2.5 (Particulate Matter 2.5 µm)",
        "folder": "notebooks/pm2_5",
        "filename": "experiment_pm2_5.ipynb",
        "target_col": "pm2_5",
    },
    {
        "key": "pm10",
        "label": "PM10 (Particulate Matter 10 µm)",
        "folder": "notebooks/pm10",
        "filename": "experiment_pm10.ipynb",
        "target_col": "pm10",
    },
    {
        "key": "tvoc",
        "label": "TVOC (Total Volatile Organic Compounds)",
        "folder": "notebooks/tvoc",
        "filename": "experiment_tvoc.ipynb",
        "target_col": "tvoc",
    },
]


def build_notebook(contaminant: dict) -> nbformat.NotebookNode:
    nb = new_notebook()

    # ── Cell 1: Header ──────────────────────────────────────────────────────
    cell_title = new_markdown_cell(f"""\
# SigKAN Experiment: {contaminant['label']}

**Senior Thesis — Systems Engineering, Universidad de Lima**

| Field | Detail |
|---|---|
| Target contaminant | `{contaminant['target_col'].upper()}` |
| Model | SigKAN Temporal |
| Authors | Rodrigo Linares and Franco Gomez |
| Institution | Universidad de Lima |

---
""")

    # ── Cell 2: Imports and path setup ──────────────────────────────────────
    cell_imports = new_code_cell("""\
import sys
import itertools

sys.path.append('../../')

import torch
import pandas as pd

print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print("Path configured.")
""")

    # ── Cell 3: Data loading and preprocessor instantiation ─────────────────
    cell_data = new_code_cell(f"""\
from src.temporal_dataset import IAQTemporalPreprocessor

DATA_PATH = '../../data/dataset.csv'
TARGET_COL = '{contaminant['target_col']}'
WINDOW_SIZE = 24   # look-back steps
HORIZON = 1        # steps ahead to predict

df = pd.read_csv(DATA_PATH)
print(f"Dataset loaded: {{df.shape[0]}} rows, {{df.shape[1]}} columns")
print(df.head())

preprocessor = IAQTemporalPreprocessor(
    target_col=TARGET_COL,
    window_size=WINDOW_SIZE,
    horizon=HORIZON,
)
dataset = preprocessor.fit_transform(df)
print(f"\\nTemporal dataset created: {{len(dataset)}} samples")
""")

    # ── Cell 4: Metrics permutations + training loop placeholder ────────────
    cell_metrics = new_code_cell(f"""\
METRICS = ['R2', 'RMSE', 'MSE', 'MAE']

permutations = list(itertools.permutations(METRICS))
print(f"Total permutations: {{len(permutations)}}\\n")
for i, perm in enumerate(permutations, 1):
    print(f"  {{i:2d}}. {{perm}}")

# ── Training loop ────────────────────────────────────────────────────────────
# TODO: instantiate SigKANTemporal and train on `dataset`
#
# from src.sigkan_temporal import SigKANTemporal
#
# results = {{}}
# for perm in permutations:
#     model = SigKANTemporal(target='{contaminant['target_col']}', metric_order=perm)
#     metrics = model.fit_evaluate(dataset)
#     results[perm] = metrics
#
# print(results)
""")

    nb.cells = [cell_title, cell_imports, cell_data, cell_metrics]

    # Minimal Jupyter metadata
    nb.metadata.update({
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {
            "name": "python",
            "version": "3.10.0",
        },
    })

    return nb


def main() -> None:
    for contaminant in CONTAMINANTS:
        out_path = os.path.join(BASE_DIR, contaminant["folder"], contaminant["filename"])
        nb = build_notebook(contaminant)
        with open(out_path, "w", encoding="utf-8") as f:
            nbformat.write(nb, f)
        print(f"[OK] {contaminant['label']:50s} → {os.path.relpath(out_path, BASE_DIR)}")

    print("\nAll notebooks generated successfully.")


if __name__ == "__main__":
    main()
