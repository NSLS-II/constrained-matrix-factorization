from constrainedmf.nmf.models import NMF
import numpy as np
import torch
from constrainedmf.utils.plotting import toy_plot

torch.manual_seed(1234)
np.random.seed(1234)


def gaussian(x, mu, sig, a):
    return a * np.exp(-np.power(x - mu, 2.0) / (2 * np.power(sig, 2.0)))


def tophat(x, s1, s2, a):
    return a * (np.heaviside(x - s1, 1) - np.heaviside(x - s2, 1))


def lorentzian(x, x0, gam, a):
    return a * gam ** 2 / (gam ** 2 + (x - x0) ** 2)


def norm_poly_weights(x, a0, a1, a2):
    this_max = np.max(a0 + x * a1 + a2 * x ** 2)
    this_min = np.min(a0 + x * a1 + a2 * x ** 2)
    return np.nan_to_num(
        ((a0 + x * a1 + a2 * x ** 2) - this_min) / (this_max - this_min)
    )


def poly_weights(x, a0, a1, a2):
    return a0 + x * a1 + a2 * x ** 2


def sigmoid(x, c, p=1):
    return 1.0 / (1.0 + np.exp(-(x - c)) ** p)


def weight_func(t):
    f1 = norm_poly_weights(t, 4, 1, -2)  # * gauss(x,5, .5, 2)
    f2 = poly_weights(t, 0.1, 0.02, -0.0002)  # * tophat(x, 3, 7, 2)
    f3 = sigmoid(t, 70, p=0.5)  # * lorentzian(x, 5, .1, 2)
    return f1, f2, f3


def merge_func(x, weights, t):
    f = weights[0][t] * gaussian(x, 6, 0.5, 2)
    f += weights[1][t] * tophat(x, 4, 6, 2)
    f += weights[2][t] * lorentzian(x, 3, 0.1, 1)
    f += np.random.normal(0, 0.01, f.shape)
    return f


def construct_overlap(n_features=1000, m_patterns=100):
    x = np.linspace(0, 10, n_features)
    y1 = gaussian(x, 6, 0.5, 2)
    y2 = tophat(x, 4, 6, 2)
    y3 = lorentzian(x, 3, 0.1, 1)
    components = np.stack([y1, y2, y3])
    t = np.arange(0, m_patterns, 1)
    weights = np.array(list(weight_func(t)))
    Y = np.matmul(weights.T, components) + np.random.normal(
        0, 0.005, (m_patterns, n_features)
    )
    return x, Y, weights, components


def standard_nmf(X):
    nmf = NMF(X.shape, n_components=3)
    nmf.fit(torch.tensor(X), beta=2)
    return nmf


def constrained_nmf(X, components):
    input_H = [
        torch.tensor(component[None, :], dtype=torch.float) for component in components
    ]
    nmf = NMF(
        X.shape,
        n_components=3,
        initial_components=input_H,
        fix_components=[True for _ in range(len(input_H))],
    )
    nmf.fit(torch.tensor(X), beta=2)
    return nmf


def model_plot(x_fun, model_fun):
    """Takes two callables, one to gen data, and one to train model on data"""

    x, Y, weights, components = x_fun()
    model = model_fun(Y, components)
    fig = toy_plot(model, x, Y, weights, components)
    return fig


def compare_figs():
    fig1 = model_plot(lambda: construct_overlap(), lambda x, comp: standard_nmf(x))
    fig2 = model_plot(
        lambda: construct_overlap(), lambda x, comp: constrained_nmf(x, comp)
    )
    return fig1, fig2


if __name__ == "__main__":
    from pathlib import Path

    path = Path(__file__).parent / "example_output"
    path.mkdir(exist_ok=True)
    fig1, fig2 = compare_figs()
    fig1.savefig(path / "standard_components.png")
    fig2.savefig(path / "constrained_components.png")
