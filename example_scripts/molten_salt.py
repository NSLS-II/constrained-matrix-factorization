from deepnmf.nmf.models import NMF
import numpy as np
import torch
from pathlib import Path
from deepnmf.companion.plotting import decomp_plot
from deepnmf.nmf.utils import sweep_components

torch.manual_seed(1234)
np.random.seed(1234)


def get_data(data_dir=None):
    if data_dir is None:
        data_dir = Path(__file__).parents[1] / "example_data/NaCl_CrCl3_pdf_ramp"
    paths = sorted(
        list(data_dir.glob("*.chi")),
        key=lambda path: float(str(path.name).split("_")[-2][1:-1]),
    )
    profiles = list()
    T = list()
    for path in paths:
        x, y = np.loadtxt(path, unpack=True)
        mask = np.logical_and(x > 0.5, x < 7.0)
        profiles.append(y[None, mask])
        T.append(float(str(path.name).split("_")[-2][1:-1]))
    X = torch.tensor(np.concatenate(profiles, axis=0), dtype=torch.float)
    T = np.array(T)
    return T, X


def standard_nmf(X):
    nmf = NMF(X.shape, 3)
    n_iter = nmf.fit(X, beta=2, tol=1e-8, max_iter=1000)
    return nmf


# def constrained_nmf(X):
# TOL = 1e-8
# MAX_ITER = 1000
# nmfs = list()
# n_components = 4
# initial_components = [torch.rand(1, X.shape[-1]) for _ in range(n_components)]
# fix_components = [False for _ in range(n_components)]
# initial_components[0] = X[0, :].reshape(1, -1)
# fix_components[0] = True
# initial_components[-1] = X[-1, :].reshape(1, -1)
# fix_components[-1] = True
# nmf = NMF(
#     X.shape,
#     n_components,
#     initial_components=initial_components,
#     fix_components=fix_components,
# )
# n_iter = nmf.fit(X, beta=2, tol=TOL, max_iter=MAX_ITER)
# nmfs.append(nmf)
#
# weight_idx = nmf.W.max(axis=0).values[1:-1].argmax() + 1
# pattern_idx = int(nmf.W.argmax(axis=0)[weight_idx])
# initial_components[weight_idx] = X[pattern_idx, :].reshape(1, -1)
# fix_components[weight_idx] = True
# nmf = NMF(
#     X.shape,
#     n_components,
#     initial_components=initial_components,
#     fix_components=fix_components,
# )
# n_iter = nmf.fit(X, beta=2, tol=TOL, max_iter=MAX_ITER)
# nmfs.append(nmf)
#
# weight_idx = 1
# pattern_idx = int(nmf.W.argmax(axis=0)[weight_idx])
# initial_components[weight_idx] = X[pattern_idx, :].reshape(1, -1)
# fix_components[weight_idx] = True
# nmf = NMF(
#     X.shape,
#     n_components,
#     initial_components=initial_components,
#     fix_components=fix_components,
# )
# n_iter = nmf.fit(X, beta=2, tol=0, max_iter=MAX_ITER * 5)
# nmfs.append(nmf)
# return nmfs


def plot_adjustments(axes):
    for ax in axes[1:]:
        ax.set_xlabel("Temperature [K]")
        ax.set_ylabel("$x_\Phi$")
        ax.set_xlim(30, 690)
    for ax in axes[2:]:
        ax.set_ylim(0, 1)
        ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
        ax.set_yticklabels([0, 0.25, 0.5, 0.75, 1])


def make_plots(n_components=4):
    from deepnmf.nmf.utils import iterative_nmf

    T, X = get_data()
    figs = list()
    axes = list()
    fig, ax = sweep_components(X, n_max=10)
    figs.append(fig),
    axes.append(ax)
    nmf = standard_nmf(X)
    fig, ax = decomp_plot(nmf, T)
    plot_adjustments(ax)
    figs.append(fig)
    axes.append(ax)
    for nmf in iterative_nmf(
        NMF, X, n_components=n_components, beta=2, tol=1e-8, max_iter=1000
    ):
        fig, ax = decomp_plot(nmf, T)
        plot_adjustments(ax)
        figs.append(fig)
        axes.append(ax)

    return figs


if __name__ == "__main__":
    path = Path(__file__).parent / "example_output"
    path.mkdir(exist_ok=True)
    figs = make_plots()
    for i, fig in enumerate(figs):
        fig.tight_layout()
        fig.show()
        fig.savefig(path / f"molten_salts_{i}.png")
