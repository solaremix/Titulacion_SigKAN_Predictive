"""
Generador de notebooks de experimentación para proyecto SigKAN + GNN.
Autores: Rodrigo Linares y Franco Gómez — Universidad de Lima
"""

import nbformat
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

CONTAMINANTES = [
    {
        "key": "co2",
        "label": "CO2 (Dióxido de Carbono)",
        "folder": "notebooks/co2",
        "filename": "experimento_co2.ipynb",
        "target_col": "co2",
    },
    {
        "key": "pm2_5",
        "label": "PM2.5 (Material Particulado 2.5 µm)",
        "folder": "notebooks/pm2_5",
        "filename": "experimento_pm2_5.ipynb",
        "target_col": "pm2_5",
    },
    {
        "key": "pm10",
        "label": "PM10 (Material Particulado 10 µm)",
        "folder": "notebooks/pm10",
        "filename": "experimento_pm10.ipynb",
        "target_col": "pm10",
    },
    {
        "key": "tvoc",
        "label": "TVOC (Compuestos Orgánicos Volátiles Totales)",
        "folder": "notebooks/tvoc",
        "filename": "experimento_tvoc.ipynb",
        "target_col": "tvoc",
    },
]


def build_notebook(contaminante: dict) -> nbformat.NotebookNode:
    nb = new_notebook()

    # ── Celda 1: Encabezado formal ──────────────────────────────────────────
    celda_titulo = new_markdown_cell(f"""\
# Experimento SigKAN-GNN: {contaminante['label']}

**Proyecto de Titulación en Ingeniería de Sistemas — Universidad de Lima**

| Campo | Detalle |
|---|---|
| Contaminante objetivo | `{contaminante['target_col'].upper()}` |
| Modelo | SigKAN + Graph Neural Network (GNN) |
| Autores | Rodrigo Linares y Franco Gómez |
| Institución | Universidad de Lima |

---
""")

    # ── Celda 2: Imports y configuración de path ────────────────────────────
    celda_imports = new_code_cell("""\
import sys
import itertools

sys.path.append('../../')

import torch
import pandas as pd

print(f"PyTorch: {torch.__version__}")
print(f"CUDA disponible: {torch.cuda.is_available()}")
print("Path configurado correctamente.")
""")

    # ── Celda 3: Carga de datos e instanciación del preprocessor ───────────
    celda_datos = new_code_cell(f"""\
from src.temporal_dataset import IAQTemporalPreprocessor

DATA_PATH = '../../data/dataset.csv'
TARGET_COL = '{contaminante['target_col']}'
WINDOW_SIZE = 24   # pasos temporales hacia atrás
HORIZON = 1        # pasos a predecir

df = pd.read_csv(DATA_PATH)
print(f"Dataset cargado: {{df.shape[0]}} filas, {{df.shape[1]}} columnas")
print(df.head())

preprocessor = IAQTemporalPreprocessor(
    target_col=TARGET_COL,
    window_size=WINDOW_SIZE,
    horizon=HORIZON,
)
dataset = preprocessor.fit_transform(df)
print(f"\\nDataset temporal creado: {{len(dataset)}} muestras")
""")

    # ── Celda 4: Permutaciones de métricas + espacio para loop ─────────────
    celda_metricas = new_code_cell(f"""\
METRICAS = ['R2', 'RMSE', 'MSE', 'MAE']

permutaciones = list(itertools.permutations(METRICAS))
print(f"Total de permutaciones posibles: {{len(permutaciones)}}\\n")
for i, perm in enumerate(permutaciones, 1):
    print(f"  {{i:2d}}. {{perm}}")

# ── Loop de entrenamiento ────────────────────────────────────────────────────
# TODO: instanciar modelo SigKAN-GNN y entrenar sobre `dataset`
#
# from src.sigkan_temporal import SigKANTemporal
#
# resultados = {{}}
# for perm in permutaciones:
#     modelo = SigKANTemporal(target='{contaminante['target_col']}', metric_order=perm)
#     metricas = modelo.fit_evaluate(dataset)
#     resultados[perm] = metricas
#
# print(resultados)
""")

    nb.cells = [celda_titulo, celda_imports, celda_datos, celda_metricas]

    # Metadata mínima para Jupyter
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
    for cont in CONTAMINANTES:
        out_path = os.path.join(BASE_DIR, cont["folder"], cont["filename"])
        nb = build_notebook(cont)
        with open(out_path, "w", encoding="utf-8") as f:
            nbformat.write(nb, f)
        print(f"[OK] {cont['label']:45s} → {os.path.relpath(out_path, BASE_DIR)}")

    print("\nTodos los notebooks generados exitosamente.")


if __name__ == "__main__":
    main()
