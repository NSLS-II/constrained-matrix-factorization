import torch
import torch.nn as nn
import torch.nn.functional as F
from nmf.metrics import Beta_divergence


def _mu_update(param, pos, gamma, l1_reg, l2_reg):
    if isinstance(param, nn.ParameterList):
        # Handle no gradients in fixed components
        grad = torch.cat([x.grad if x.requires_grad else torch.zeros_like(x) for x in param])
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
    def __init__(self, W_shape, H_shape,
                 n_components,
                 initial_components=None,
                 fix_components=()):
        """
        Base class for setting up NMF
        W is (m_examples, n_components)
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
        self.fix_neg = nn.Threshold(0., 1e-8)
        self.rank = n_components

        W = torch.rand(*W_shape)
        self.W = nn.Parameter(W)
        if initial_components is not None:
            l = [nn.Parameter(component) for component in initial_components]
        else:
            l = [nn.Parameter(torch.rand(1, *H_shape[1:])) for _ in range(H_shape[0])]
        if fix_components:
            for i in range(len(fix_components)):
                l[i].requires_grad = not fix_components[i]
        self.H_list = nn.ParameterList()
        for param in l:
            self.H_list.append(param)

    def forward(self, H_list=None, W=None):
        if H_list is None:
            H_list = self.H_list
        if W is None:
            W = self.W
        H = torch.cat([x for x in H_list])
        return self.reconstruct(H, W)

    def reconstruct(self, H, W):
        raise NotImplementedError

    def get_W_positive(self, WH, beta, H_sum) -> (torch.Tensor, None or torch.Tensor):
        raise NotImplementedError

    def get_H_positive(self, WH, beta, W_sum) -> (torch.Tensor, None or torch.Tensor):
        raise NotImplementedError

    def fit(self,
            X,
            update_W=True,
            update_H=True,
            beta=1,
            tol=1e-5,
            max_iter=200,
            alpha=0,
            l1_ratio=0):

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
            H = torch.cat([x for x in self.H_list])
            if self.W.requires_grad:
                self.zero_grad()
                WH = self.reconstruct(H.detach(), self.W)
                loss = Beta_divergence(self.fix_neg(WH), X, beta)
                loss.backward()

                with torch.no_grad():
                    positive_comps, H_sum = self.get_W_positive(WH, beta, H_sum)
                    _mu_update(self.W, positive_comps, gamma, l1_reg, l2_reg)
                W_sum = None

            if any([x.requires_grad for x in self.H_list]):
                self.zero_grad()
                WH = self.reconstruct(H, self.W.detach())
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
        return n_iter, self.W


class NMF(NMFBase):

    def __init__(self, X_shape, n_components, **kwargs):
        self.m_examples, self.n_features = X_shape
        super().__init__((self.m_examples, n_components), (n_components, self.n_features), n_components, **kwargs)

    def reconstruct(self, H, W):
        return W @ H

    def get_W_positive(self, WH, beta, H_sum):
        H = torch.cat([x for x in self.H_list])
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
        _, maxidx = self.W.data.max(0)
        _, idx = maxidx.sort()
        self.W.data = self.W.data[:, idx]
        for H in self.H_list:
            H.data = H.data[idx]

class NMFD(NMFBase):

    def __init__(self, X_shape, n_components, T=1, **kwargs):
        self.m_examples, self.n_features = X_shape
        self.pad_size = T-1
        super().__init__((self.m_examples, n_components, T), (n_components, self.n_features - T + 1), n_components, **kwargs)

    def reconstruct(self, H, W):
        return F.conv1d(H[None, :], W.flip(2), padding=self.pad_size)[0]

    def get_W_positive(self, WH, beta, H_sum):
        H = torch.cat([x for x in self.H_list])
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
        _, maxidx = self.W.data.sum(2).max(0)
        _, idx = maxidx.sort()
        self.W.data = self.W.data[:, idx]
        for H in self.H_list:
            H.data = H.data[idx]

if __name__ == "__main__":
    from nmf.models import NMF
    import numpy as np
    import torch


    def gaussian(x, mu, sig):
        return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))


    n_features = 1000
    m_patterns = 10
    x = np.linspace(-10, 10, n_features)
    y1 = gaussian(x, 0, 1) + np.random.normal(0, .01, (m_patterns, n_features))
    y2 = gaussian(x, -5, 1) + np.random.normal(0, .01, (m_patterns, n_features))
    y3 = gaussian(x, 5, 1) + np.random.normal(0, .01, (m_patterns, n_features))
    Y = np.concatenate((y1, y2, y3), axis=0)
    fixin = gaussian(x, 0.1, 1)
    input_H = (torch.tensor(fixin, dtype=torch.float).reshape(1, n_features),
               torch.rand(1, n_features),
               torch.rand(1, n_features))
    model = NMFD(Y.shape, 3, initial_components=input_H, T=1, fix_components=(True, False, False))
    #model = NMFD(Y.shape,3, T=5)
    n_iter = model.fit(torch.tensor(Y), beta=2, alpha=0.5)
