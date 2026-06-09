from __future__ import annotations

from typing import Any

import torch


def load_legacy_torch(path: str, *, map_location: Any = "cpu"):
    """Load legacy torch artifacts saved with pickle-backed objects.

    PyTorch 2.6 defaults to weights_only=True, which breaks loading Data objects,
    sampler batches, and other legacy pickled artifacts used by this repo.
    """

    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        # Older PyTorch versions do not accept the weights_only kwarg.
        return torch.load(path, map_location=map_location)
