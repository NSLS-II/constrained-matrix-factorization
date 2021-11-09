from constrainedmf.nmf.models import NMF
import torch


def test_standard_nmf(linearly_mixed_gaussians):
    """Test standard NMF using Euclidian loss and KL-divergence loss"""
    xs, weights, components = linearly_mixed_gaussians
    nmf = NMF(xs.shape, n_components=2)
    # Ensure optimization reaches a reasonable loss
    nmf.fit(torch.tensor(xs), beta=1, max_iter=500)
    assert nmf.loss(xs, beta=1) < 1
    assert nmf.loss(xs, beta=2) < 0.1
    nmf = NMF(xs.shape, n_components=2)
    nmf.fit(torch.tensor(xs), beta=2, max_iter=500)
    assert nmf.loss(xs, beta=1) < 10.0
    assert nmf.loss(xs, beta=2) < 1.0
    # Ensure weights vary somewhat linearly by asserting max and min are end members
    weights = nmf.W
    assert set(int(x) for x in weights.max(dim=0).indices) == {20, 0}
    assert set(int(x) for x in weights.min(dim=0).indices) == {20, 0}


def test_constrained_weights(linearly_mixed_gaussians):
    """Assure constrained weights are constrained"""
    xs, weights, components = linearly_mixed_gaussians
    initial_weights = [w[None, :] for w in weights.T]
    nmf = NMF(
        xs.shape,
        n_components=2,
        initial_weights=initial_weights,
        fix_weights=[True for _ in range(len(initial_weights))],
    )
    nmf.fit(xs)
    assert torch.all(torch.eq(weights.T, nmf.W))


def test_constrained_components(linearly_mixed_gaussians):
    """Assure constrained components are constrained"""
    xs, weights, components = linearly_mixed_gaussians
    initial_components = [component[None, :] for component in components]
    nmf = NMF(
        xs.shape,
        n_components=2,
        initial_components=initial_components,
        fix_components=[True for _ in range(len(initial_components))],
    )
    nmf.fit(xs)
    assert torch.all(torch.eq(components, nmf.H))
