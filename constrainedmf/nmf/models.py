import torch
import torch.nn as nn
import torch.nn.functional as F
from constrainedmf.nmf.metrics import Beta_divergence


def _mu_update(param, pos, gamma, l1_reg, l2_reg):
    """
    Perform multiplicative update of param (W, or H)

    Parameters
    ----------
    param: tensor
        Weights or components
    pos: tensor
        positive denominator (mu)
    gamma: float
        Beta - 1 from Beta divergence loss
    l1_reg: float
        L1 regularization
    l2_reg: float
        L2 regularization

    Returns
    -------

    """
    if isinstance(param, nn.ParameterList):
        # Handle no gradients in fixed components
        grad = torch.cat(
            [x.grad if x.requires_grad else torch.zeros_like(x) for x in param]
        )
    elif param.grad is None:
        return
    else:
        grad = param.grad
    # prevent negative terms and zero division
    multiplier = F.relu(pos - grad, inplace=True)
    if (pos == 0).sum() > 0:
        pos.add_(1e-7)

    if l1_reg > 0:
        pos.add_(l1_reg)
    if l2_reg > 0:
        if isinstance(param, nn.ParameterList):
            reg_param = torch.cat([x for x in param])
        else:
            reg_param = param
        if pos.shape != reg_param.shape:
            pos = pos + l2_reg * reg_param
        else:
            pos.add_(l2_reg * reg_param)

    multiplier.div_(pos)
    if gamma != 1:
        multiplier.pow_(gamma)
    if isinstance(param, nn.ParameterList):
        for i, sub_param in enumerate(param):
            sub_param.mul_(multiplier[i, :])
    else:
        param.mul_(multiplier)


class NMFBase(nn.Module):
    def __init__(
        self,
        W_shape,
        H_shape,
        n_components,
        *,
        initial_components=None,
        fix_components=(),
        initial_weights=None,
        fix_weights=(),
        device=None,
        **kwargs
    ):
        """
        Base class for setting up NMF

        Parameters
        ----------
        W_shape: tuple of int
            Shape of the weights matrix
        H_shape: tuple of int
            Shape of the components matrix
        n_components: int
            Number of components in the factorization
        initial_components: tuple of torch.Tensor
            Initial components for the factorization. Shape (1, n_features)
        fix_components: tuple of bool
            Corresponding directive to fix each component in the factorization.
            The components are ordered, and the default behavior is to allow a component to vary.
            I.e. (True, False, True) for a 4 component factorization will result in the first and third
            component being fixed, while the second and fourth vary.
        initial_weights: tuple of torch.Tensor
            Initial weights for the factorization. Shape (1, m_examples)
        fix_weights: tuple of bool
            Corresponding directive to fix each weight in the factorization.
        device: str, torch.device, None
            Device for matrix factorization to proceed on. Defaults to cpu.
        kwargs: dict
            Keyword arguments for torch.nn.Module

        """
        super().__init__()
        self.fix_neg = nn.Threshold(0.0, 1e-8)
        self.rank = n_components
        if device is None:
            self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)

        if initial_weights is not None:
            w_list = [nn.Parameter(weight) for weight in initial_weights] + [
                nn.Parameter(torch.rand(1, *W_shape[1:]))
                for _ in range(W_shape[0] - len(initial_weights))
            ]
        else:
            w_list = [
                nn.Parameter(torch.rand(1, *W_shape[1:])) for _ in range(W_shape[0])
            ]
        if fix_weights:
            for i in range(len(fix_weights)):
                w_list[i].requires_grad = not fix_weights[i]
        self.W_list = nn.ParameterList(w_list).to(device)

        if initial_components is not None:
            h_list = [nn.Parameter(component) for component in initial_components] + [
                nn.Parameter(torch.rand(1, *H_shape[1:]))
                for _ in range(H_shape[0] - len(initial_components))
            ]
        else:
            h_list = [
                nn.Parameter(torch.rand(1, *H_shape[1:])) for _ in range(H_shape[0])
            ]
        if fix_components:
            for i in range(len(fix_components)):
                h_list[i].requires_grad = not fix_components[i]
        self.H_list = nn.ParameterList(h_list).to(device)

    @property
    def H(self):
        return torch.cat([x for x in self.H_list])

    @property
    def W(self):
        return torch.cat([x for x in self.W_list])

    def loss(self, X, beta=2):
        with torch.no_grad():
            WH = self.reconstruct(self.H, self.W)
            return Beta_divergence(self.fix_neg(WH), X, beta)

    def forward(self, H=None, W=None):
        if H is None:
            H = self.H
        if W is None:
            W = self.W
        return self.reconstruct(H, W)

    def reconstruct(self, H, W):
        """
        Method for reconstructing the approximate input matrix from the components and weights

        Parameters
        ----------
        H: torch.Tensor
            Components matrix
        W: torch.Tensor
            Weights matrix

        Returns
        -------

        """
        raise NotImplementedError

    def get_W_positive(self, WH, beta, H_sum) -> (torch.Tensor, None or torch.Tensor):
        """
        Get the positive denominator an/or H sum for multiplicative W update

        Parameters
        ----------
        WH: torch.Tensor
            Reconstruction of input matrix (in the simple case this is the matrix produce W @ H
        beta: int, float
            Value for beta divergence
        H_sum: torch.Tensor, None
            Sum over components matrix to use in denominator of update. If unknown or not required use None.

        Returns
        -------

        """
        raise NotImplementedError

    def get_H_positive(self, WH, beta, W_sum) -> (torch.Tensor, None or torch.Tensor):
        """
        Get the positive denominator and/or W sum for multiplicative H update

        Parameters
        ----------
        WH: torch.Tensor
            Reconstruction of input matrix (in the simple case this is the matrix produce W @ H
        beta: int, float
            Value for beta divergence
        W_sum: torch.Tensor, None
            Sum over weights matrix to use in denominator of update. If unknown or not required use None.

        Returns
        -------

        """
        raise NotImplementedError

    def fit(
        self,
        X,
        update_W=True,
        update_H=True,
        beta=1,
        tol=1e-5,
        max_iter=200,
        alpha=0,
        l1_ratio=0,
    ):
        """
        Fit the wights (W) and components (H) to the dataset X.

        Parameters
        ----------
        X: torch.Tensor
            Tensor of the dataset to fit, shape (m_examples, n_features)
        update_W: bool
            Override on updating weights matrix
        update_H: bool
            Override on updating components matrix
        beta: float
            Value for beta divergence
        tol: float
            Change in loss tolerance for exiting optimization loop
        max_iter: int
            Maximum number of iterations to consider for optimization loop
        alpha: float
            Amount of regularization for the mu update
        l1_ratio: float
            Ratio of L1 to L2 regularization

        Returns
        -------

        """

        X = X.type(torch.float).to(self.device)
        X = self.fix_neg(X)

        if beta < 1:
            gamma = 1 / (2 - beta)
        elif beta > 2:
            gamma = 1 / (beta - 1)
        else:
            gamma = 1

        l1_reg = alpha * l1_ratio
        l2_reg = alpha * (1 - l1_ratio)

        loss_scale = torch.prod(torch.tensor(X.shape)).float()
        losses = []

        H_sum, W_sum = None, None

        if max_iter < 1:
            raise ValueError("Maximum number of iterations must be at least 1.")

        for n_iter in range(max_iter):
            # W update
            if update_W and any([x.requires_grad for x in self.W_list]):
                self.zero_grad()
                WH = self.reconstruct(self.H.detach(), self.W)
                loss = Beta_divergence(self.fix_neg(WH), X, beta)
                loss.backward()

                with torch.no_grad():
                    positive_comps, H_sum = self.get_W_positive(WH, beta, H_sum)
                    _mu_update(self.W_list, positive_comps, gamma, l1_reg, l2_reg)
                W_sum = None

            # H update
            if update_H and any([x.requires_grad for x in self.H_list]):
                self.zero_grad()
                WH = self.reconstruct(self.H, self.W.detach())
                loss = Beta_divergence(self.fix_neg(WH), X, beta)
                loss.backward()

                with torch.no_grad():
                    positive_comps, W_sum = self.get_H_positive(WH, beta, W_sum)
                    _mu_update(self.H_list, positive_comps, gamma, l1_reg, l2_reg)
                H_sum = None

            loss = loss.div_(loss_scale).item()

            if not n_iter:
                loss_init = loss
            elif (previous_loss - loss) / loss_init < tol:  # noqa: F821
                break
            previous_loss = loss  # noqa:F841
            losses.append(loss)

        return losses

    def fit_transform(self, *args, **kwargs):
        self.fit(*args, **kwargs)
        return self.W


class NMF(NMFBase):
    def __init__(
        self,
        X_shape,
        n_components,
        *,
        initial_components=None,
        fix_components=(),
        initial_weights=None,
        fix_weights=(),
        device=None,
        **kwargs
    ):
        """
        Standard NMF with ability for constraints constructed from input matrix shape.

        W is (m_examples, n_components)

        H is (n_components, n_example_features)

        W @ H give reconstruction of X.

        Parameters
        ----------
        X_shape: tuple
            Tuple of ints describing shape of input matrix
        n_components: int
            Number of desired components for the matrix factorization
        initial_components: tuple of torch.Tensor
            Initial components for the factorization. Shape (1, n_features)
        fix_components: tuple of bool
            Corresponding directive to fix each component in the factorization.
            The components are ordered, and the default behavior is to allow a component to vary.
            I.e. (True, False, True) for a 4 component factorization will result in the first and third
            component being fixed, while the second and fourth vary.
        initial_weights: tuple of torch.Tensor
            Initial weights for the factorization. Shape (1, m_examples)
        fix_weights: tuple of bool
            Corresponding directive to fix each weight in the factorization.
        device: str, torch.device, None
            Device for matrix factorization to proceed on. Defaults to cpu.
        kwargs: dict
            kwargs for torch.nn.Module
        """
        self.m_examples, self.n_features = X_shape
        super().__init__(
            (self.m_examples, n_components),
            (n_components, self.n_features),
            n_components,
            initial_components=initial_components,
            initial_weights=initial_weights,
            fix_weights=fix_weights,
            fix_components=fix_components,
            device=device,
            **kwargs
        )

    def reconstruct(self, H, W):
        """
        Reconstructs the approximate input matrix from matrix product of weights and components

        Parameters
        ----------
        H: torch.Tensor
            Components matrix
        W: torch.Tensor
            Weights matrix

        Returns
        -------
        torch.Tensor

        """
        return W @ H

    def get_W_positive(self, WH, beta, H_sum):
        H = self.H
        if beta == 1:
            if H_sum is None:
                H_sum = H.sum(1)
            denominator = H_sum[None, :]
        else:
            if beta != 2:
                WH = WH.pow(beta - 1)
            WHHt = WH @ H.t()
            denominator = WHHt

        return denominator, H_sum

    def get_H_positive(self, WH, beta, W_sum):
        W = self.W
        if beta == 1:
            if W_sum is None:
                W_sum = W.sum(0)  # shape(n_components, )
            denominator = W_sum[:, None]
        else:
            if beta != 2:
                WH = WH.pow(beta - 1)
            WtWH = W.t() @ WH
            denominator = WtWH
        return denominator, W_sum

    def sort(self):
        raise NotImplementedError


class NMFD(NMFBase):
    """
    Deconvolutional NMF
    W is (m_examples, n_components, kernel_width)

    H is (n_components, n_example_features)
    """

    def __init__(self, X_shape, n_components, T=1, **kwargs):
        self.m_examples, self.n_features = X_shape
        self.pad_size = T - 1
        super().__init__(
            (self.m_examples, n_components, T),
            (n_components, self.n_features - T + 1),
            n_components,
            **kwargs
        )

    def reconstruct(self, H, W):
        return F.conv1d(H[None, :], W.flip(2), padding=self.pad_size)[0]

    def get_W_positive(self, WH, beta, H_sum):
        H = self.H
        if beta == 1:
            if H_sum is None:
                H_sum = H.sum(1)
            denominator = H_sum[None, :, None]
        else:
            if beta != 2:
                WH = WH.pow(beta - 1)
            WHHt = F.conv1d(WH[:, None], H[:, None])
            denominator = WHHt

        return denominator, H_sum

    def get_H_positive(self, WH, beta, W_sum):
        W = self.W
        if beta == 1:
            if W_sum is None:
                W_sum = W.sum((0, 2))
            denominator = W_sum[:, None]
        else:
            if beta != 2:
                WH = WH.pow(beta - 1)
            WtWH = F.conv1d(WH[None, :], W.transpose(0, 1))[0]
            denominator = WtWH
        return denominator, W_sum

    def sort(self):
        raise NotImplementedError
