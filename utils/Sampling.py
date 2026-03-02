import numpy as np

def get_abar(S, A, Phi, theta):
    """
    Compute greedy policy for a given parameter and feature set
    """
    abar = np.zeros(S, dtype=int)
    for s in range(S):
        abar[s] = int(np.argmax(Phi[s*A:(s+1)*A,:].dot(theta))) # note: tie breaking according to NumPy's defaults
    return abar

def get_stationary_dist(P, max_iter=10000, tol=1e-12):
    """
    This is for small MDPs only. For larger MDPs please use other methods
    """
    S = P.shape[0]
    d = np.ones(S) / S

    for _ in range(max_iter):
        d_next = d @ P
        if np.linalg.norm(d_next - d, ord=1) < tol:
            break
        d = d_next

    return d


