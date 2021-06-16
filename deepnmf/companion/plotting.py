import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import torch


def toy_plot(model, x, Y, weights, components):
    fig = plt.figure(tight_layout=True, figsize=(15, 10))
    gs = GridSpec(2, 6)
    recon_axes = [fig.add_subplot(gs[0, i * 2 : (i + 1) * 2]) for i in range(3)]
    comp_ax = fig.add_subplot(gs[1, :3])
    weight_ax = fig.add_subplot(gs[1, 3:])

    with torch.no_grad():
        recon = model.reconstruct(model.H, model.W)

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

    H = model.H.data.numpy()
    ax = comp_ax
    for i in range(H.shape[0]):
        ax.plot(x, H[i, :], label=f"Learned Component {i}")
        ax.plot(x, components[i, :], "--k", label=f"True Component {i}")
    ax.set_title("Learned Components")
    ax.legend()

    W = model.W.data.numpy()
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
