import pytest
import numpy as np
import torch


def gaussian(x, mu, sig):
    return np.exp(-np.power(x - mu, 2.0) / (2 * np.power(sig, 2.0)))


@pytest.fixture
def linearly_mixed_gaussians():
    """
    Dataset of 21 linearly mixed gaussians, blended from start to finish
    Returns
    -------
    ys: tensor
        Dataset shape (21, 1000)
    weights: tensor
        Linearly varying weights, shape (2, 21)
    components: tensor
        Original components, shape (2, 1000)

    """
    # Dataset of 21 linearly mixed gaussians, blended from start to finish
    x = np.linspace(-10, 10, 1000)
    y1 = gaussian(x, 1.2, 1)
    y2 = gaussian(x, -1 * 1.2, 1)
    weights = np.array(list(zip(*[(x, 1 - x) for x in np.linspace(0, 1, 21)])))
    ys = np.matmul(weights.T, np.array([y1, y2]))
    return (
        torch.tensor(ys, dtype=torch.float),
        torch.tensor(weights, dtype=torch.float),
        torch.tensor(np.stack([y1, y2], axis=0), dtype=torch.float),
    )
