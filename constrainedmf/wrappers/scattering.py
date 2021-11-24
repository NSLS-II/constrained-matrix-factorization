"""Non-negative matrix factorization of full datasets for total scattering experiments (XRD, PDF)"""
from constrainedmf.nmf.models import NMF, NMFD
from constrainedmf.nmf.utils import iterative_nmf
import torch
import numpy as np


def _decomposition_preprocess(*, Q, I, q_range, bkg_removal, normalize):  # noqa: E741
    """Preprocess options for trimming q-range, removing background, and normalization"""
    if q_range is None:
        idx_min = 0
        idx_max = I.shape[1]
    else:
        idx_min = (
            np.where(Q[0, :] < q_range[0])[0][-1]
            if len(np.where(Q[0, :] < q_range[0])[0])
            else 0
        )
        idx_max = (
            np.where(Q[0, :] > q_range[1])[0][0]
            if len(np.where(Q[0, :] > q_range[1])[0])
            else I.shape[1]
        )

    sub_I = I[:, idx_min:idx_max]
    sub_Q = Q[:, idx_min:idx_max]

    # Data manipulation
    if bkg_removal:
        import peakutils

        bases = []
        for i in range(sub_I.shape[0]):
            bases.append(peakutils.baseline(sub_I[i, :], deg=bkg_removal))
        bases = np.stack(bases)
        sub_I = sub_I - bases
    if normalize:
        sub_I = (sub_I - np.min(sub_I, axis=1, keepdims=True)) / (
            np.max(sub_I, axis=1, keepdims=True) - np.min(sub_I, axis=1, keepdims=True)
        )

    # Numerical stability of non-negativity
    if np.min(sub_I) < 0:
        sub_I = sub_I - np.min(sub_I, axis=1, keepdims=True)

    return sub_Q, sub_I, idx_min, idx_max


def decomposition(
    Q,
    I,  # noqa: E741
    *,
    n_components=3,
    q_range=None,
    initial_components=None,
    fix_components=(),
    mode="Linear",
    kernel_width=1,
    max_iter=1000,
    bkg_removal=None,
    normalize=False,
    device=None,
    **kwargs,
):
    """
    Decompose and label a set of I(Q) data with optional focus bounds. Can be used for other
    1-dimensional response functions, written with total scattering in mind.

    Two operating modes are available: Linear (conventional) and Deconvolutional. The former will proceed as conventional
    NMF as implemented in sklearn, with the added flexibility of the torch implementation. The latter will include a
    convolutional kernel in the reconstruction between the component and weight matricies.

    Initial components can be set as starting conditions of the component matrix for the optimization. These components
    can be fixed or allowed to vary using the `fix_components` argument as a tuple of booleans.

    Keyword arguments are passed to the fit method

    Parameters
    ----------
    Q : array
        Ordinate Q for I(Q). Assumed to be rank 2, shape (m_patterns, n_data)
    I : array
        The intensity values for each Q, assumed to be the same shape as Q. (m_patterns, n_data)
    n_components: int
        Number of components for NMF
    q_range : tuple, list
        (Min, Max) Q values for consideration in NMF. This enables a focused region for decomposition.
    initial_components: array
        Initial starting conditions of intensity components. Assumed to be shape (n_components, n_data).
        If q_range is given, these will be trimmed in accordance with I.
    fix_components: tuple(bool)
        Flags for fixing a subset of initial components
    mode: {"Linear", "Deconvolutional"}
        Operating mode
    kernel_width: int
        Width of 1-dimensional convolutional kernel
    max_iter: int
        Maximum number of iterations for NMF
    bkg_removal: int, optional
        Integer degree for peakutils background removal
    normalize: bool, optional
        Flag for min-max normalization
    device: str, torch.device, None
            Device for matrix factorization to proceed on. Defaults to cpu.
    **kwargs: dict
        Arguments passed to the fit method. See nmf.models.NMFBase.

    Returns
    -------
    sub_Q : array
        Subsampled ordinate used for NMF
    sub_I : array
        Subsampled I used for NMF
    alphas : array
        Resultant weights from NMF
    components:  array
        Resultant components from NMF

    """

    sub_Q, sub_I, idx_min, idx_max = _decomposition_preprocess(
        Q=Q, I=I, q_range=q_range, bkg_removal=bkg_removal, normalize=normalize
    )

    # Initial components
    if mode != "Deconvolutional":
        kernel_width = 1
    n_features = sub_I.shape[1]
    if initial_components is None:
        input_H = None
    else:
        input_H = []
        for i in range(n_components):
            try:
                sub_H = initial_components[i][idx_min:idx_max]
                sub_H = sub_H[kernel_width // 2 : len(sub_H) - kernel_width // 2 + 1]
                if normalize:
                    sub_H = (sub_H - np.min(sub_H)) / (np.max(sub_H) - np.min(sub_H))
                input_H.append(
                    torch.tensor(sub_H, dtype=torch.float).reshape(
                        1, n_features - kernel_width + 1
                    )
                )
            except IndexError:
                input_H.append(torch.rand(1, n_features - kernel_width + 1))

    # Model construction
    if mode == "Linear":
        model = NMF(
            sub_I.shape,
            n_components,
            initial_components=input_H,
            fix_components=fix_components,
            device=device,
        )
    elif mode == "Deconvolutional":
        model = NMFD(
            sub_I.shape,
            n_components,
            T=kernel_width,
            initial_components=input_H,
            fix_components=fix_components,
            device=device,
        )
    else:
        raise NotImplementedError

    _, W = model.fit_transform(torch.tensor(sub_I), max_iter=max_iter, **kwargs)

    if len(W.shape) > 2:
        alphas = torch.mean(W, 2).data.numpy()
    else:
        alphas = W.data.numpy()

    components = torch.cat([x for x in model.H_list]).data.numpy()
    return sub_Q, sub_I, alphas, components


def iterative_decomposition(
    Q,
    I,  # noqa: E741
    *,
    n_components=3,
    q_range=None,
    mode="Linear",
    kernel_width=1,
    bkg_removal=None,
    normalize=False,
    **kwargs,
):
    """
    Iterative decomposition by performing constrained NMF using dataset members as constraints.
    The first 2 constraints are the end members of the dataset.
    The next constraint is chosen by examining the most prominent (heavily weighted) learned component.
    The index where this component's weight is the highest, is used to select the next constraint from the dataset.
    This continues until all components are constrained by dataset members.

    Parameters
    ----------
    Q : array
        Ordinate Q for I(Q). Assumed to be rank 2, shape (m_patterns, n_data)
    I : array
        The intensity values for each Q, assumed to be the same shape as Q. (m_patterns, n_data)
    n_components: int
        Number of components for NMF
    q_range : tuple, list
        (Min, Max) Q values for consideration in NMF. This enables a focused region for decomposition.
    mode: {"Linear", "Deconvolutional"}
        Operating mode
    kernel_width: int
        Width of 1-dimensional convolutional kernel
    bkg_removal: int, optional
        Integer degree for peakutils background removal
    normalize: bool, optional
        Flag for min-max normalization
    kwargs: dict
        Keyword arguments first get passed to nmf.utils.iterative_nmf to control fit parameters.
        Fit parameters include beta, alpha, tol, max_iter.
        Keyword arguments are then passed to the class initialization.
        Common init parameters include device, or initial_weights

    Returns
    -------
    sub_Q: array
        Subselected region of Q
    sub_I: array
        Subselected region of I
    weights: array
        Final weights from NMF
    components: array
        Final components from NMF

    """
    sub_Q, sub_I, idx_min, idx_max = _decomposition_preprocess(
        Q=Q, I=I, q_range=q_range, bkg_removal=bkg_removal, normalize=normalize
    )

    if mode == "Deconvolutional":
        nmf_class = NMFD
    elif mode == "Linear":
        nmf_class = NMF
    else:
        raise NotImplementedError(f"Mode {mode} unavailable.")

    # Safety check
    if "initial_components" in kwargs:
        del kwargs["initial_components"]
    if "fix_components" in kwargs:
        del kwargs["fix_components"]

    nmfs = iterative_nmf(
        nmf_class,
        torch.tensor(sub_I, dtype=torch.float),
        n_components=n_components,
        kernel_width=kernel_width,
        **kwargs,
    )

    nmf = nmfs[-1]
    W = nmf.W
    H = nmf.H
    if len(W.shape) > 2:
        weights = torch.mean(W, 2).data.numpy()
    else:
        weights = W.data.numpy()
    components = H.data.numpy()
    return sub_Q, sub_I, weights, components
