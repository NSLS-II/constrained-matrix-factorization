import torch
import torch.nn as nn
import torch.nn.functional as F
from deepnmf.nmf.metrics import Beta_divergence


def _mu_update(param, pos, gamma, l1_reg, l2_reg):
    if isinstance(param, nn.ParameterList):
        # Handle no gradients in fixed components
        grad = torch.cat(
            [x.grad if x.requires_grad else torch.zeros_like(x) for x in param]
        )
    elif param.grad is None:
        return
    else:
        grad = param.grad
    # prevent negative term, very likely to happen with kl divergence
    multiplier = F.relu(pos - grad, inplace=True)

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
        initial_components=None,
        fix_components=(),
        initial_weights=None,
        fix_weights=(),
        **kwargs
    ):
        """
        Base class for setting up NMF
        W is (m_examples, n_components) or (m_examples, n_components, kernel_width)
        H is (n_components, n_example_features)
        W @ H give reconstruction of X.
        Parameters
        ----------
        W_shape
        H_shape
        n_components
        initial_components
        fix_components
        """
        super().__init__()
        self.fix_neg = nn.Threshold(0.0, 1e-8)
        self.rank = n_components

        W = torch.rand(*W_shape)
        self.W = nn.Parameter(W)
        if initial_weights is not None:
            w_list = [nn.Parameter(weight) for weight in initial_weights]
        else:
            w_list = [
                nn.Parameter(torch.rand(1, *W_shape[1:])) for _ in range(W_shape[0])
            ]
        if fix_weights:
            for i in range(len(fix_weights)):
                w_list[i].requires_grad = not fix_weights[i]
        self.W_list = nn.ParameterList(w_list)

        if initial_components is not None:
            h_list = [nn.Parameter(component) for component in initial_components]
        else:
            h_list = [
                nn.Parameter(torch.rand(1, *H_shape[1:])) for _ in range(H_shape[0])
            ]
        if fix_components:
            for i in range(len(fix_components)):
                h_list[i].requires_grad = not fix_components[i]
        self.H_list = nn.ParameterList(h_list)

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
        raise NotImplementedError

    def get_W_positive(self, WH, beta, H_sum) -> (torch.Tensor, None or torch.Tensor):
        raise NotImplementedError

    def get_H_positive(self, WH, beta, W_sum) -> (torch.Tensor, None or torch.Tensor):
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

        X = X.type(torch.float)
        X = self.fix_neg(X)
        # self.W.requires_grad = update_W
        # self.H.requires_grad = update_H

        if beta < 1:
            gamma = 1 / (2 - beta)
        elif beta > 2:
            gamma = 1 / (beta - 1)
        else:
            gamma = 1

        l1_reg = alpha * l1_ratio
        l2_reg = alpha * (1 - l1_ratio)

        loss_scale = torch.prod(torch.tensor(X.shape)).float()

        H_sum, W_sum = None, None

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
            elif (previous_loss - loss) / loss_init < tol:
                break
            previous_loss = loss

        return n_iter

    def fit_transform(self, *args, **kwargs):
        n_iter = self.fit(*args, **kwargs)
        return n_iter, torch.cat([x for x in self.W_list])


class NMF(NMFBase):
    def __init__(self, X_shape, n_components, **kwargs):
        self.m_examples, self.n_features = X_shape
        super().__init__(
            (self.m_examples, n_components),
            (n_components, self.n_features),
            n_components,
            **kwargs
        )

    def reconstruct(self, H, W):
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
