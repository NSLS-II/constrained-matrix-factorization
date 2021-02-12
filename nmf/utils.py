import torch


def normalize(x: torch.Tensor, axis=0) -> torch.Tensor:
    return x / x.sum(axis, keepdim=True)
