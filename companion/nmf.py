"Non-negative matrix factorization of full datasets"
from nmf.models import NMF, NMFD
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl


def decomposition(
    Q,
    I,
    n_components=3,
    q_range=None,
    initial_components=None,
    fix_components=(),
    mode="Linear",
    kernel_width=1,
    max_iter=1000,
    bkg_removal=None,
    normalize=False,
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

    # Data subselection
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
                sub_H = sub_H[kernel_width // 2 : -kernel_width // 2 + 1]
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
        )
    elif mode == "Deconvolutional":
        model = NMFD(
            sub_I.shape,
            n_components,
            T=kernel_width,
            initial_components=input_H,
            fix_components=fix_components,
        )
    else:
        raise NotImplementedError

    _, W = model.fit_transform(torch.Tensor(sub_I), max_iter=max_iter, **kwargs)

    if len(W.shape) > 2:
        alphas = torch.mean(W, 2).data.numpy()
    else:
        alphas = W.data.numpy()

    components = torch.cat([x for x in model.H_list]).data.numpy()
    return sub_Q, sub_I, alphas, components


def waterfall(ax, xs, ys, alphas, color="k", sampling=1, offset=0.2, **kwargs):
    indicies = range(0, xs.shape[0])[::sampling]
    for plt_i, idx in enumerate(indicies):
        y = ys[idx, :] + plt_i * offset
        x = xs[idx, :]
        ax.plot(x, y, color=color, alpha=alphas[idx], **kwargs)
    return ax


def example_plot(
    sub_Q,
    sub_I,
    alphas,
    axes=None,
    sax=None,
    components=None,
    comax=None,
    cmap="tab10",
    alt_ordinate=None,
    offset=1.0,
    summary_fig=False,
):
    """
    Example plotting of NMF results. Not necessarily for Bluesky deployment

    Parameters
    ----------
    sub_Q: array
        Q to plot in I(Q)
    sub_I: array
        I to plot in I(Q)
    alphas: array
        transparencies of multiple repeated plots of I(Q)
    axes: optional existing axes for waterfalls
    sax: optional axes for summary figure
    cmap: mpl colormap
    alt_ordinate: array
        Array len sub_I.shape[0], corresponding to an alternative labeled dimension for which to order the stacked plots
    summary_fig: bool
        Whether to include separate figure of alphas over the ordinate

    Returns
    -------
    fig, axes

    """

    n_components = alphas.shape[1]
    cmap = mpl.cm.get_cmap(cmap)
    norm = mpl.colors.Normalize(vmin=0, vmax=n_components)

    # Create alternative ordinate for the waterfall/stacking
    if not alt_ordinate is None:
        idxs, labels = list(
            zip(*sorted(zip(range(sub_I.shape[0]), alt_ordinate), key=lambda x: x[1]))
        )
    else:
        idxs = list(range(sub_I.shape[0]))
        labels = list(range(sub_I.shape[0]))
    xs = sub_Q[idxs, :]
    ys = sub_I[idxs, :]
    alphas = alphas[idxs, :]

    # Order by proxy center of mass of class in plot regime. Makes the plots feel like a progression not random.
    alpha_ord = np.argsort(np.matmul(np.arange(alphas.shape[0]), alphas))

    if axes is None:
        fig, axes = plt.subplots(
            int(np.ceil(np.sqrt(n_components))), int(np.ceil(np.sqrt(n_components)))
        )
        axes = axes.reshape(-1)
    else:
        axes = np.ravel(axes)
    for i, ax in enumerate(axes):
        if i < n_components:
            i_a = alpha_ord[i]
            color = cmap(norm(i))
            alpha = (alphas[:, i_a] - np.min(alphas[:, i_a])) / (
                np.max(alphas[:, i_a]) - np.min(alphas[:, i_a])
            )
            ax = waterfall(ax, xs, ys, alpha, color=color, offset=offset)
        else:
            ax.set_visible = False

    if summary_fig:
        if sax is None:
            sfig, sax = plt.subplots(figsize=(6, 6))

        sx = np.arange(0, alphas.shape[0])
        for i in range(alphas.shape[1]):
            sax.plot(
                sx,
                alphas[:, alpha_ord[i]],
                color=cmap(norm(i)),
                label=f"Component {i + 1}",
            )

    if components is not None:
        if comax is None:
            comfig, comax = plt.subplots(figsize=(6, 6))
        for i in range(components.shape[0]):
            kernel_width = xs.shape[1] - components.shape[1] + 1
            comax.plot(
                xs[0][kernel_width // 2 : -kernel_width // 2 + 1],
                components[i, :] + i,
                color=cmap(norm(i)),
            )

    return
