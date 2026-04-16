"""
Quick smoke-test: instantiate SigKANTemporal and run a forward pass.
Usage: uv run python main.py
"""

import torch
from src import SigKANTemporal, IAQ_TARGETS

BATCH    = 4
WINDOW   = 24
FEATURES = 10   # placeholder feature count


def main() -> None:
    model = SigKANTemporal(
        in_features=FEATURES,
        window=WINDOW,
        hidden_sizes=[64, 32],
        out_features=len(IAQ_TARGETS),
        num_basis=8,
        num_mix_layers=2,
    )

    x = torch.randn(BATCH, WINDOW, FEATURES)
    pred   = model(x)
    latent = model.encode(x)

    print(f"SigKANTemporal — parameters: {model.count_parameters():,}")
    print(f"Input:   {tuple(x.shape)}")
    print(f"Output:  {tuple(pred.shape)}  → {IAQ_TARGETS}")
    print(f"Latent:  {tuple(latent.shape)}")
    print("Smoke-test OK")


if __name__ == "__main__":
    main()
