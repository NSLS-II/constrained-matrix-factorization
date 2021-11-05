import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.gridspec import GridSpec
import numpy as np
import torch


def toy_plot(model, x, Y, weights, components):
    fig = plt.figure(tight_layout=True, figsize=(15, 10))
    gs = GridSpec(2, 6)
    recon_axes = [fig.add_subplot(gs[0, i * 2 : (i + 1) * 2]) for i in range(3)]
    comp_ax = fig.add_subplot(gs[1, :3])
    weight_ax = fig.add_subplot(gs[1, 3:])

    with torch.no_grad():
        recon = model.reconstruct(model.H, model.W).cpu()

    ax = recon_axes[0]
    ax.plot(x, Y[0, :], label="Truth")
    ax.plot(x, recon[0, :].data.numpy(), label="Reconstruct")
    ax = recon_axes[1]
    ax.plot(x, Y[Y.shape[0] // 2, :], label="Truth")
    ax.plot(x, recon[recon.shape[0] // 2, :].data.numpy(), label="Reconstruct")
    ax = recon_axes[2]
    ax.plot(x, Y[-1, :], label="Truth")
    ax.plot(x, recon[-1, :].data.numpy(), label="Reconstruct")
    for ax in recon_axes:
        ax.legend()

    H = model.H.data.cpu()
    ax = comp_ax
    for i in range(H.shape[0]):
        ax.plot(x, H[i, :], label=f"Learned Component {i}")
        ax.plot(x, components[i, :], "--k", label=f"True Component {i}")
    ax.set_title("Learned Components")
    ax.legend()

    W = model.W.data.cpu()
    ax = weight_ax
    for i in range(W.shape[1]):
        ax.plot(W[:, i], label=f"Learned Weights {i}")
        ax.plot(weights.T[:, i], "--k", label=f"True Weights {i}")
    ax.set_title("Learned Weights")
    ax.legend()
    return fig


def decomp_plot(nmf, T, axes=None):
    H = nmf.H.data.numpy()
    W = nmf.W.data.numpy()
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    ax = axes[0]
    for i in range(H.shape[0]):
        ax.plot(H[i, :] / H[i, :].max() + i)
    ax.set_title("Stacked Normalized Components")

    ax = axes[1]
    for i in range(W.shape[1]):
        ax.plot(T, W[:, i])
    ax.set_title("Weights")

    ax = axes[2]
    for i in range(W.shape[1]):
        ax.plot(T, W[:, i] / W[:, i].max())
    ax.set_title("Normalized Weights")
    return fig, axes


def waterfall(ax, xs, ys, alphas, color="k", sampling=1, offset=0.2, **kwargs):
    indicies = range(0, xs.shape[0])[::sampling]
    for plt_i, idx in enumerate(indicies):
        y = ys[idx, :] + plt_i * offset
        x = xs[idx, :]
        ax.plot(x, y, color=color, alpha=alphas[idx], **kwargs)
    return ax


def summary_plot(
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
    if alt_ordinate is not None:
        idxs, labels = list(
            zip(*sorted(zip(range(sub_I.shape[0]), alt_ordinate), key=lambda x: x[1]))
        )
    else:
        idxs = list(range(sub_I.shape[0]))

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
            waterfall(ax, xs, ys, alpha, color=color, offset=offset)
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
            if kernel_width == 1:
                comax.plot(
                    xs[0][:],
                    components[alpha_ord[i], :] + i,
                    color=cmap(norm(i)),
                )
            else:
                comax.plot(
                    xs[0][kernel_width // 2 : -kernel_width // 2 + 1],
                    components[alpha_ord[i], :] + i,
                    color=cmap(norm(i)),
                )

    return
