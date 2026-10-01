"""Rebuild the NumPy export from bundled, checksum-verified Summit tensors.

Optional maintenance utility: requires torch, which the evaluator does not need.
Uses weights_only=True; downloads nothing and performs no training.
"""

import hashlib
import json
from pathlib import Path

import numpy as np
import torch


def main():
    here = Path(__file__).resolve().parent
    manifest = json.loads((here / "UPSTREAM_MANIFEST.json").read_text())
    for entry in manifest["files"]:
        assert hashlib.sha256((here / entry["retained_file"]).read_bytes()).hexdigest() == entry["sha256"]
    weights = {}
    for index in range(5):
        state = torch.load(here / f"data/baumgartner_aniline_cn_crosscoupling_predictor_{index}.pt",
                           map_location="cpu", weights_only=True)
        for name, value in state.items():
            weights[f"{index}.{name}"] = value.detach().numpy()
    np.savez_compressed(here / "data/weights.npz", **weights)
    print("Exported the five frozen networks; no training was performed.")


if __name__ == "__main__":
    main()
