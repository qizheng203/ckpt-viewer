"""Create a tiny synthetic checkpoint for manual testing."""

from __future__ import annotations

from pathlib import Path

import torch


def main() -> None:
    out = Path(__file__).with_name("tiny_checkpoint.pth")
    checkpoint = {
        "epoch": 1,
        "model": {
            "module.conv.weight": torch.randn(8, 3, 3, 3),
            "module.conv.bias": torch.randn(8),
            "module.bn.weight": torch.ones(8),
            "module.bn.bias": torch.zeros(8),
            "module.bn.running_mean": torch.zeros(8),
            "module.bn.running_var": torch.tensor([1e-8, 1e-5, 1e-4, 0.1, 0.2, 0.3, 0.4, 1.0]),
            "module.bn.num_batches_tracked": torch.tensor(5, dtype=torch.int64),
            "module.fc.weight": torch.randn(10, 128),
            "module.fc.bias": torch.randn(10),
        },
        "optimizer": {},
    }
    torch.save(checkpoint, out)
    print(out)


if __name__ == "__main__":
    main()
