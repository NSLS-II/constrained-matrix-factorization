from constrainedmf.nmf.models import NMF
import numpy as np
import torch
from constrainedmf.utils.plotting import toy_plot

torch.manual_seed(1234)
np.random.seed(1234)
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def gaussian(x, mu, sig):
    return np.exp(-np.power(x - mu, 2.0) / (2 * np.power(sig, 2.0)))


def construct_overlap(n_features=1000, m_patterns=21, mu=1.2, sig=1):
    x = np.linspace(-10, 10, n_features)
    y1 = gaussian(x, mu, sig)
    y2 = gaussian(x, -1 * mu, sig)
    weights = np.array(list(zip(*[(x, 1 - x) for x in np.linspace(0, 1, m_patterns)])))
    weights += np.random.normal(0, 0.01, weights.shape)
    Y = np.matmul(weights.T, np.array([y1, y2])) + np.random.normal(
        0, 0.01, (m_patterns, n_features)
    )
    return x, Y, weights, np.stack([y1, y2], axis=0)


def standard_nmf(X):
    nmf = NMF(X.shape, n_components=2, device=DEVICE)
    nmf.fit(torch.tensor(X), beta=2)
    return nmf


def constrained_nmf(X):
    ideal_weights = np.array(
        list(zip(*[(x, 1 - x) for x in np.linspace(0, 1, X.shape[0])]))
    )
    input_W = [torch.tensor(w[None, :], dtype=torch.float) for w in ideal_weights.T]
    nmf = NMF(
        X.shape,
        n_components=2,
        initial_weights=input_W,
        fix_weights=[True for _ in range(len(input_W))],
        device=DEVICE,
    )
    nmf.fit(torch.tensor(X), beta=2)
    return nmf


def model_plot(x_fun, model_fun):
    """Takes two callables, one to gen data, and one to train model on data"""

    x, Y, weights, components = x_fun()
    model = model_fun(Y)
    fig = toy_plot(model, x, Y, weights, components)
    return fig


def compare_figs():
    fig1 = model_plot(lambda: construct_overlap(), standard_nmf)
    fig2 = model_plot(lambda: construct_overlap(), constrained_nmf)
    return fig1, fig2


if __name__ == "__main__":
    from pathlib import Path

    path = Path(__file__).parent / "example_output"
    path.mkdir(exist_ok=True)
    fig1, fig2 = compare_figs()
    fig1.savefig(path / "standard_weights.png")
    fig2.savefig(path / "constrained_weights.png")
