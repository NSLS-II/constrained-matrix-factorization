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


def standard_nmf(X, n_components=4):
    nmf = NMF(X.shape, n_components)
    nmf.fit(X, beta=2, tol=1e-8, max_iter=1000)
    return nmf


def plot_adjustments(axes):
    for ax in axes[1:]:
        ax.set_xlabel("Temperature [K]")
        ax.set_ylabel("$x_\Phi$")  # noqa: W605
        ax.set_xlim(30, 690)
    for ax in axes[2:]:
        ax.set_ylim(0, 1)
        ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
        ax.set_yticklabels([0, 0.25, 0.5, 0.75, 1])


def make_plots(n_components=4):
    from constrainedmf.nmf.utils import iterative_nmf

    T, X = get_data()
    figs = list()
    axes = list()
    fig, ax = sweep_components(X, n_max=10)
    figs.append(fig),
    axes.append(ax)
    nmf = standard_nmf(X, n_components)
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
    for n in (3, 4, 5):
        figs = make_plots(n)
        for i, fig in enumerate(figs):
            fig.tight_layout()
            fig.show()
            fig.savefig(path / f"molten_salts_{n}_{i}.png")
