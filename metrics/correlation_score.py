from typing import Callable

import numpy as np
import scipy
import torch
from torch import nn


def display_scores(results: list[float]) -> tuple[float, float, float]:
    mean = np.mean(results)
    std = np.std(results, ddof=1)
    sem = scipy.stats.sem(results)
    conf_interval = sem * scipy.stats.t.ppf(
        (1 + 0.95) / 2.0, len(results) - 1
    )  # 95% confidence interval
    return round(mean, 4), round(std, 4), round(conf_interval, 4)


def random_choice(size: int, num_select: int = 100) -> np.ndarray:
    select_idx = np.random.randint(low=0, high=size, size=(num_select,))
    return select_idx


def cacf_torch(
    x: torch.Tensor, max_lag: int, dim: tuple[int, ...] = (0, 1)
) -> torch.Tensor:
    def get_lower_triangular_indices(n):
        return [list(x) for x in torch.tril_indices(n, n)]

    ind = get_lower_triangular_indices(x.shape[2])
    x = (x - x.mean(dim, keepdims=True)) / x.std(dim, keepdims=True)
    n = x.shape[2]

    if n > 1:
        return _cacf_torch_chunked(x, max_lag, dim, chunk_size=50)

    else:
        x_l = x[..., ind[0]]
        x_r = x[..., ind[1]]
        cacf_list = list()
        for i in range(max_lag):
            y = x_l[:, i:] * x_r[:, :-i] if i > 0 else x_l * x_r
            cacf_i = torch.mean(y, (1))
            cacf_list.append(cacf_i)
        cacf = torch.cat(cacf_list, 1)
        return cacf.reshape(cacf.shape[0], -1, len(ind[0]))


def _cacf_torch_chunked(
    x: torch.Tensor, max_lag: int, dim: tuple[int, ...] = (0, 1), chunk_size: int = 1000
) -> torch.Tensor:
    """
    Chunked approach to avoid creating huge (B, T, ~51k) tensors at once
    when n == 321.
    """

    def get_lower_triangular_indices(n):
        return [list(x) for x in torch.tril_indices(n, n)]

    ind = get_lower_triangular_indices(x.shape[2])
    B = x.shape[0]
    total_indices = len(ind[0])

    cacf_lag_results = []
    for lag in range(max_lag):
        chunk_outputs = []
        start = 0
        while start < total_indices:
            end = min(start + chunk_size, total_indices)
            # Gather only a slice of the lower-triangular indices
            chunk_0 = ind[0][start:end]
            chunk_1 = ind[1][start:end]

            x_l = x[..., chunk_0]
            x_r = x[..., chunk_1]

            if lag > 0:
                y = x_l[:, :, lag:] * x_r[:, :, :-lag]
            else:
                y = x_l * x_r

            # Mean over time dimension => shape: (B, chunk_size)
            cacf_i_chunk = torch.mean(y, dim=1)
            chunk_outputs.append(cacf_i_chunk)

            start = end

        # For this lag, concatenate => shape: (B, total_indices)
        lag_result = torch.cat(chunk_outputs, dim=1)
        cacf_lag_results.append(lag_result)

    # Combine all lags => (B, total_indices * max_lag)
    cacf_full = torch.cat(cacf_lag_results, dim=1)
    # Reshape => (B, max_lag, total_indices)
    return cacf_full.reshape(B, max_lag, total_indices)


class Loss(nn.Module):
    def __init__(
        self,
        name: str,
        reg: float = 1.0,
        transform: Callable[[torch.Tensor], torch.Tensor] = lambda x: x,
        threshold: float = 10.0,
        backward: bool = False,
        norm_foo: Callable[[torch.Tensor], torch.Tensor] = lambda x: x,
    ):
        super(Loss, self).__init__()
        self.name = name
        self.reg = reg
        self.transform = transform
        self.threshold = threshold
        self.backward = backward
        self.norm_foo = norm_foo

    def forward(self, x_fake: torch.Tensor) -> torch.Tensor:
        self.loss_componentwise = self.compute(x_fake)
        return self.reg * self.loss_componentwise.mean()

    def compute(self, x_fake: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError()

    @property
    def success(self) -> torch.Tensor:
        return torch.all(self.loss_componentwise <= self.threshold)


class CrossCorrelLoss(Loss):
    def __init__(self, x_real: torch.Tensor, **kwargs):
        super(CrossCorrelLoss, self).__init__(
            norm_foo=lambda x: torch.abs(x).sum(0), **kwargs
        )
        self.cross_correl_real = cacf_torch(self.transform(x_real), 1).mean(0)[0]

    def compute(self, x_fake: torch.Tensor) -> torch.Tensor:
        cross_correl_fake = cacf_torch(self.transform(x_fake), 1).mean(0)[0]
        loss = self.norm_foo(
            cross_correl_fake - self.cross_correl_real.to(x_fake.device)
        )
        return loss / 10.0


def calculate_pearson_correlation(
    real_sig: np.ndarray, gen_sig: np.ndarray
) -> tuple[float, float, float]:
    iterations = 1

    x_real = torch.from_numpy(real_sig)
    x_fake = torch.from_numpy(gen_sig)

    correlational_score = []
    size = int(x_real.shape[0] / iterations)

    for i in range(iterations):
        real_idx = random_choice(x_real.shape[0], size)
        fake_idx = random_choice(x_fake.shape[0], size)
        corr = CrossCorrelLoss(x_real[real_idx, :, :], name="CrossCorrelLoss")
        loss = corr.compute(x_fake[fake_idx, :, :])
        correlational_score.append(loss.item())
        print(f"Iter {i}: ", "cross-correlation =", loss.item(), "\n")

    mean, std, conf_interval = display_scores(correlational_score)

    return mean, std, conf_interval
