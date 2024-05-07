import numpy as np
import matplotlib.pyplot as plt
from numpy.typing import NDArray
from typing import Tuple, Dict
from scipy.optimize import minimize

def safe_inv(v: NDArray):
    return np.divide(np.ones_like(v), v, out=np.zeros_like(v), where=(v!=0))

# Adapted from https://github.com/gabrielmoryoussef/VertexHunting
def SuccessiveProj(R: NDArray, K: int) -> NDArray:
    # Points in R^n
    n = R.shape[0]
    r = R.shape[1]
    vertices = np.zeros((K, r))
    Y = np.column_stack((np.ones(n), R))
    
    for i in range(K):
        ind = np.argmax(np.sum(Y**2, axis=1))
        vertices[i, :] = R[ind, :]
        u = Y[ind, :] / np.sqrt(np.sum(Y[ind, :]**2))
        length = np.dot(Y, u)
        Y -= np.outer(length, u)
    
    return vertices

# For step: compute W_hat
def w_star_obj(q: NDArray, d: NDArray, B: NDArray) -> np.float64:
    S = B @ q
    return np.sum((d - S)**2) + (1-np.sum(q))**2

def w_star_minimizer(d: NDArray, B: NDArray) -> Tuple[NDArray, np.float64]:
    r = B.shape[1]
    q = np.zeros(r)
    result = minimize(w_star_obj, q, args=(d, B))
    return result.x, result.fun

# For step: normalize W_hat
def normalize_columns(P: NDArray, norm='l2') -> NDArray:
    for i in range(P.shape[1]):
        if np.all(P[:, i] == 0):
            raise ValueError(f'Column {i} is all zeros')
    if norm == 'l1':
        return P / np.abs(P).sum(axis=0)
    if norm == 'l2':
        return P / np.linalg.norm(P, axis=0)
    raise ValueError('norm must be either l1 or l2')

def estimate_decomposition(N: NDArray, r: int) -> Dict:
    if N.ndim != 2:
        raise ValueError("Matrix must be 2D")
    if N.shape[0] != N.shape[1]:
        raise ValueError("Matrix must be square")
    if N.dtype not in ('int64', 'int32'):
        raise ValueError("Matrix must contain integers")
    if not np.all(N >= 0):
        raise ValueError("Counts must be non-negative")
    if np.isnan(N).any():
        raise ValueError("Matrix contains NaN")
    if np.isinf(N).any():
        raise ValueError("Matrix contains inf")
    p = N.shape[0]

    N_tilde = N @ np.diag(safe_inv(np.sqrt(N.T @ np.ones(p))))
    _, _, h_hats = np.linalg.svd(N_tilde)
    h_hats = h_hats.T
    D_hat = np.diag(safe_inv(h_hats[:, 0])) @ h_hats[:, 1:r]
    b_hat = SuccessiveProj(D_hat, r).T

    W_hat_star = [w_star_minimizer(D_hat[i,:], b_hat)[0] for i in range(p)]
    W_hat_star = np.array([np.where(x < 0, 0, x) for x in W_hat_star])
    W_hat = normalize_columns(W_hat_star.T, 'l1').T

    V_hat = np.diag(h_hats[:, 0]) @ np.diag(np.sqrt(N.T @ np.ones(p))) @ W_hat
    V_hat = np.abs(normalize_columns(V_hat, 'l1'))
    P_hat = np.diag(safe_inv(N @ np.ones(p))) @ N
    U_hat = P_hat @ V_hat @ np.linalg.inv(V_hat.T @ V_hat)
    anchor_states = [
        list(np.where(np.isclose(W_hat[:, meta_state], 1.0))[0])
        for meta_state in range(r)
    ]
    return {
        'p': p,
        'r': r,
        'N': N,
        'P_hat': P_hat,
        'H_hat': h_hats,
        'D_hat': D_hat,
        'b_hat': b_hat,
        'W_hat': W_hat,
        'V_hat': V_hat,
        'U_hat': U_hat,
        'anchor states': anchor_states
    }