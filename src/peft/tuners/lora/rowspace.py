import os
from dataclasses import dataclass
from typing import Literal

import torch
from torch import nn


@dataclass
class PivotedQRState:
    type: Literal["pivotedQR"]
    # These does need to be saved
    W2: torch.Tensor
    W_bar: torch.Tensor

    # Only need to save these
    # Can recalculate the above using this
    # and the original weight
    sel: torch.Tensor
    rest: torch.Tensor


def computeWbarAndW2PivotedQR(matrix: torch.Tensor, cache_path: str | None = None) -> PivotedQRState | None:
    n, m = matrix.shape
    if n == m:
        return None

    def computeWbarAndW2Wide(matrix: torch.Tensor) -> PivotedQRState | None:
        from scipy.linalg import qr

        k = matrix.shape[0]
        orig_dtype = matrix.dtype
        orig_device = matrix.device
        W = matrix.detach().cpu().numpy()

        R, P = qr(W, pivoting=True, mode="r")
        R = torch.as_tensor(R, dtype=orig_dtype)
        P = torch.as_tensor(P, dtype=torch.long)
        R11 = R[:, :k]
        W_bar = torch.linalg.solve_triangular(R11, R[:, k:], upper=True)

        return PivotedQRState(
            type="pivotedQR",
            W2=matrix[:, P[:k]].contiguous(),
            W_bar=W_bar.to(device=orig_device).contiguous(),
            sel=P[:k].to(device=orig_device).contiguous(),
            rest=P[k:].to(device=orig_device).contiguous(),
        )

    FILE_NAME = "rowspace_pivotedQR_cache.bin"
    FILE_PATH = os.path.join(cache_path, FILE_NAME) if cache_path is not None else None
    if FILE_PATH:
        cached = torch.load(
            FILE_PATH,
            map_location="cpu",
            weights_only=True,
        )
        sel: torch.Tensor = cached["sel"]
        rest: torch.Tensor = cached["rest"]
        if n < m:
            # Wide
            W2 = matrix[:, sel].contiguous()
            X = torch.linalg.solve(W2, matrix[:, rest])
        else:
            W2 = matrix[sel, :].contiguous()
            X = torch.linalg.solve(W2.T, matrix[rest, :].T).T
        return PivotedQRState("pivotedQR", W2, X, torch.tensor(sel), torch.tensor(rest))

    # Build from scratch
    if n < m:
        return computeWbarAndW2Wide(matrix)
    result = computeWbarAndW2Wide(matrix.T)
    if result is None:
        return result
    if FILE_PATH:
        os.makedirs(cache_path)
        file_path = os.path.join(cache_path, FILE_NAME)
        torch.save(
            {
                "sel": result.sel,
                "rest": result.rest,
            },
            file_path,
        )
    result.W2 = result.W2.T.contiguous()
    result.W_bar = result.W_bar.T.contiguous()
    return result


@dataclass
class SVDState:
    type: Literal["SVD"]
    W2: torch.Tensor
    P: torch.Tensor


def computeWbarAndW2SVD(matrix: torch.Tensor) -> SVDState | None:
    n, m = matrix.shape
    if n == m:
        return None
    if n < m:
        U, S, Vh = torch.linalg.svd(matrix, full_matrices=False)
        return SVDState(type="SVD", W2=U * S, P=Vh)
    else:
        U, S, Vh = torch.linalg.svd(matrix, full_matrices=False)
        return SVDState(type="SVD", W2=S[:, None] * Vh, P=U)


@dataclass
class NaiveState:
    type: Literal["naive"]
    W2: torch.Tensor
    W_bar: torch.Tensor


def computeWbarAndW2Naive(matrix: torch.Tensor) -> NaiveState | None:
    n, m = matrix.shape
    if n == m:
        return None
    if n < m:
        W1 = matrix[:, : m - n]
        W2 = matrix[:, m - n :]
        W_bar = torch.linalg.solve(W2, W1)
    else:
        W1 = matrix[: n - m, :]
        W2 = matrix[n - m :, :]
        W_bar = torch.linalg.solve(W2.T, W1.T).T
    return NaiveState("naive", W2, W_bar)


def forwardNaive(state: NaiveState, lora_A: nn.Linear, lora_B: nn.Linear, x: torch.Tensor) -> torch.Tensor:
    # Apply lora to W2
    W2_with_lora = state.W2 + lora_A @ lora_B

    n, m = state.W_bar.shape
    if n < m:
        pass
    else:
        pass
