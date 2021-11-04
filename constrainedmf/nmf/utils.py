import torch
import numpy as np
import scipy.stats as st


def normalize(x: torch.Tensor, axis=0) -> torch.Tensor:
    return x / x.sum(axis, keepdim=True)


def scalar_to_vec(x, k, dist="unif", nsig=3):
    """
    Helper function to generate 1d kernel distributions that integrate
      to a specified scalar. Used in generalizing NMF initialization
      weights to NMFD.
    ------------
    Parameters
    ------------
    x: float to which the 1d kernel should integrate
    k: integer width of the 1d kernel
    dist: probability distribution ("unif" or "gauss") to use for kernel
    nsig: for Gaussian kernels, number of SDs to span in k steps
    ------------
    Returns
    ------------
    numpy array of shape [1,k]
    """
    if dist == "unif":
        return np.ones(k) * (x / k)

    elif dist == "gauss":
        t = np.linspace(-nsig, nsig, k + 1)
        return np.diff(st.norm.cdf(t)) * x

    else:
        raise ValueError('Currently supports only "unif" and "gauss"')


def sweep_components(X, n_max=None, n_min=2):
    """
    Sweeps over all values of n_components and returns a plot of losses vs n_components

    Parameters
    ----------
    X : Tensor
    n_max : int
        Max n in search, default X.shape[0]
    n_min : int
        Min n in search, default 2

    Returns
    -------
    fig, axes
    """
    import matplotlib.pyplot as plt
    from constrainedmf.nmf.models import NMF

    if n_max is None:
        n_max = X.shape[0]

    losses = list()
    kl_losses = list()
    for n_components in range(n_min, n_max + 1):
        nmf = NMF(X.shape, n_components)
        nmf.fit(X, beta=2, tol=1e-8, max_iter=500)
        losses.append(nmf.loss(X, beta=2))
        kl_losses.append(nmf.loss(X, beta=1))

    fig, axes = plt.subplots(1, 2)
    x = list(range(n_min, n_max + 1))
    axes[0].plot(x, losses, label="MSE Loss")
    axes[0].set_title("MSE Loss")
    axes[1].plot(x, kl_losses, label="KL Loss")
    axes[1].set_title("KL Loss")
    fig.suptitle("Loss vs # of Components")
    return fig, axes


def iterative_nmf(
    NMFClass, X, n_components, *, beta=2, alpha=0.0, tol=1e-8, max_iter=1000, **kwargs
):
    """
    Utility for performing NMF on a stream of data along a common state variable
    (Temperature or composition), that coincides with the data ordering.

    Parameters
    ----------
    NMFClass : class
        Child class of NMFBase
    X : Tensor
        Data to perform NMF on
    n_components : int
        Number of components for NMF
    beta : int
        Beta for determining loss function
    alpha : float
        Alpha for determining regularization. Default 0.0 is no regularization.
    tol : float
        Optimization tolerance
    max_iter : int
        Maximum optimization iterations
    kwargs : dict
        Passed to intialization of NMF

    Returns
    -------
    nmfs : list of NMF instances

    """
    nmfs = list()
    initial_components = [torch.rand(1, X.shape[-1]) for _ in range(n_components)]
    fix_components = [False for _ in range(n_components)]

    # Start on bounding the outer components
    initial_components[0] = X[0, :].reshape(1, -1)
    fix_components[0] = True
    initial_components[-1] = X[-1, :].reshape(1, -1)
    fix_components[-1] = True

    nmf = NMFClass(
        X.shape,
        n_components,
        initial_components=initial_components,
        fix_components=fix_components,
        **kwargs
    )
    nmf.fit(X, beta=beta, tol=tol, max_iter=max_iter, alpha=alpha)
    nmfs.append(nmf)

    if len(nmf.W.shape) == 3:
        convolutional = True
    else:
        convolutional = False
    visited = {0, n_components - 1}
    for _ in range(n_components - 2):

        # Find next most prominent weight
        if convolutional:
            indices = (
                nmf.W.sum(axis=-1).max(axis=0).values.argsort(descending=True).numpy()
            )
        else:
            indices = nmf.W.max(axis=0).values.argsort(descending=True).numpy()
        for i in indices:
            if i in visited:
                continue
            else:
                visited.add(i)
                weight_idx = i
                break

        # Find most important component to that weight
        if convolutional:
            pattern_idx = int(nmf.W.sum(axis=-1).argmax(axis=0)[weight_idx])
        else:
            pattern_idx = int(nmf.W.argmax(axis=0)[weight_idx])

        # Lock and run
        initial_components[weight_idx] = X[pattern_idx, :].reshape(1, -1)
        fix_components[weight_idx] = True
        nmf = NMFClass(
            X.shape,
            n_components,
            initial_components=initial_components,
            fix_components=fix_components,
            **kwargs
        )
        nmf.fit(X, beta=beta, tol=tol, max_iter=max_iter, alpha=alpha)
        nmfs.append(nmf)

    return nmfs
