from constrainedmf.nmf.models import NMF
import numpy as np
import torch
from pathlib import Path
from constrainedmf.utils.plotting import decomp_plot
from constrainedmf.nmf.utils import sweep_components

torch.manual_seed(1234)
np.random.seed(1234)


def get_data(data_dir=None):
    if data_dir is None:
        data_dir = Path("~/Datasets/HTPXRD/BaTiO/exp_patterns").expanduser()
    if data_dir is None:
        data_dir = Path(__file__).parents[1] / "example_data/NaCl_CrCl3_pdf_ramp"
    paths = sorted(list(data_dir.glob("*.npy")))
    profiles = list()
    T = list()
    for path in paths:
        x = np.load(path)
        T.append(
            float(str(path).split("_")[2][:3])
        )  # hardcoded nonsense for T from label
        profiles.append(x / x.max())
    X = torch.tensor(np.concatenate(profiles, axis=1).T, dtype=torch.float)
    T = np.array(T)
    return T, X


def standard_nmf(X):
    nmf = NMF(X.shape, 4)
    nmf.fit(X, beta=2, tol=1e-8, max_iter=1000)
    return nmf


def plot_adjustments(axes):
    for ax in axes[1:]:
        ax.axvline(185, linestyle="--", color="k")
        ax.axvline(280, linestyle="--", color="k")
        ax.axvline(400, linestyle="--", color="k")
        ax.set_ylim(0, 1)
        ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
        ax.set_yticklabels([0, 0.25, 0.5, 0.75, 1])
        ax.set_xlabel("Temperature [K]")
        ax.set_ylabel("$x_\Phi$")  # noqa: W605
        ax.set_xlim(150, 445)


def make_plots():
    from constrainedmf.nmf.utils import iterative_nmf

    T, X = get_data()
    figs = list()
    axes = list()
    fig, ax = sweep_components(X, n_max=8)
    figs.append(fig),
    axes.append(ax)
    fig, ax = decomp_plot(standard_nmf(X), T)
    plot_adjustments(ax)
    figs.append(fig)
    axes.append(ax)
    for nmf in iterative_nmf(NMF, X, n_components=4, beta=2, tol=1e-8, max_iter=1000):
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
        fig.savefig(path / f"BaTiO_{i}.png")
