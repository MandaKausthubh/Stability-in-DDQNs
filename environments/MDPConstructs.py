"""
MDP constructions from "Does DQN Learn?" paper by Gopalan & Thoppe.
These are the specific MDP examples used to demonstrate DQN's pathological behaviors.
"""
import numpy as np
from environments.DiscreteMDPs import DiscreteMDP


def create_figure2_mdp():
    """
    Create the 2-state, 2-action MDP from Figure 2 of the paper.
    This MDP demonstrates three different behaviors:
    - Convergence to optimal policy (blue trajectory)
    - Convergence to sub-optimal policy (green trajectory)
    - Policy oscillation (red trajectory - sliding mode on boundary)

    Returns:
        DiscreteMDP: The MDP instance
        np.ndarray: Feature matrix Phi of shape (SA, d) = (4, 2)
        float: Discount factor gamma
        float: Exploration parameter epsilon
    """
    # Number of states and actions
    S, A = 2, 2

    # Transition probabilities P(s'|s,a)
    # P[a][s,s'] = P(s'|s,a)
    P = np.zeros((S, A, S))

    # Action 1 transitions
    P[:, 0, :] = np.array([
        [0.380, 0.620],
        [0.786, 0.214]
    ])

    # Action 2 transitions
    P[:, 1, :] = np.array([
        [0.124, 0.876],
        [0.426, 0.574]
    ])

    # Reward vector r(s,a) - expected reward for each state-action pair
    # Paper gives: r = [-0.031, 0.785, -0.282, -0.418]
    # This appears to be r(s,a) = E[r(s,a,s')] = sum_{s'} P(s'|s,a) * r(s,a,s')
    # But we need the full r(s,a,s') for our MDP
    # From the paper, it seems r is given as r(s,a) directly
    # We'll use expected rewards r(s,a)
    r_expected = np.array([
        [-0.031, 0.785],   # r(s=0, a=0), r(s=0, a=1)
        [-0.282, -0.418]   # r(s=1, a=0), r(s=1, a=1)
    ])

    # For the DiscreteMDP class, we need r(s,a,s')
    # We'll construct this assuming r(s,a,s') = r(s,a) for all s'
    # This gives E[r(s,a,s')] = r(s,a) as required
    r = np.zeros((S, A, S))
    for s in range(S):
        for a in range(A):
            for s_next in range(S):
                r[s, a, s_next] = r_expected[s, a]

    # Create MDP with uniform initial distribution
    rho = np.ones(S) / S
    mdp = DiscreteMDP(S, A, P, r, rho)

    # Feature matrix Phi from paper (4x2 matrix)
    # Each row corresponds to phi(s,a), ordered as (s=0,a=0), (s=0,a=1), (s=1,a=0), (s=1,a=1)
    Phi = np.array([
        [1.919, 0.112],
        [2.581, -0.659],
        [1.912, 1.679],
        [1.560, -0.168]
    ])

    gamma = 0.9
    epsilon = 0.05

    return mdp, Phi, gamma, epsilon


def create_figure3b_mdp():
    """
    Create the 2-state, 2-action MDP from Figure 3b of the paper.
    This MDP demonstrates the worst-case scenario where linear DQN
    always converges to the worst possible policy.

    Returns:
        DiscreteMDP: The MDP instance
        np.ndarray: Feature matrix Phi of shape (SA, d) = (4, 2)
        float: Discount factor gamma
        float: Exploration parameter epsilon
    """
    S, A = 2, 2

    # Transition probabilities
    P = np.zeros((S, A, S))

    # Action 1 transitions
    P[:, 0, :] = np.array([
        [0.355, 0.645],
        [0.598, 0.402]
    ])

    # Action 2 transitions
    P[:, 1, :] = np.array([
        [0.820, 0.180],
        [0.288, 0.712]
    ])

    # Reward vector from paper
    r_expected = np.array([
        [-0.599, -1.427],   # r(s=0, a=0), r(s=0, a=1)
        [0.658, 0.300]      # r(s=1, a=0), r(s=1, a=1)
    ])

    r = np.zeros((S, A, S))
    for s in range(S):
        for a in range(A):
            for s_next in range(S):
                r[s, a, s_next] = r_expected[s, a]

    rho = np.ones(S) / S
    mdp = DiscreteMDP(S, A, P, r, rho)

    # Feature matrix Phi from paper
    Phi = np.array([
        [0.985, 0.951],
        [0.395, 1.078],
        [-0.904, 1.276],
        [0.063, 1.214]
    ])

    gamma = 0.75
    epsilon = 0.1

    return mdp, Phi, gamma, epsilon


def compute_babar(P, r, Phi, epsilon, gamma, num_states, num_actions):
    """
    Compute b_{\bar{a}} and A_{\bar{a}} matrices for a deterministic policy \bar{a}.

    These are defined in equations (13) and (14) of the paper:
    b_{\bar{a}} = E[\phi(s,a) r(s,a,s')] = \Phi^T D^{\epsilon}_{\bar{a}} r
    A_{\bar{a}} = E[\phi(s,a) \phi^T(s,a) - \gamma \phi(s,a) \phi^T(s',a')]
               = \Phi^T D^{\epsilon}_{\bar{a}} (I - \gamma P^{\epsilon'}_{\bar{a}}) \Phi

    Args:
        P: Transition matrix of shape (S, A, S)
        r: Reward matrix of shape (S, A, S)
        Phi: Feature matrix of shape (SA, d)
        epsilon: Exploration parameter
        gamma: Discount factor
        num_states: Number of states
        num_actions: Number of actions

    Returns:
        dict: Dictionary mapping policy (as tuple) to (b_a, A_a) matrices
    """
    from itertools import product

    S, A = num_states, num_actions
    d = Phi.shape[1]

    results = {}

    # Enumerate all deterministic policies
    for policy in product(range(A), repeat=S):
        policy = np.array(policy)

        # Build epsilon-greedy policy pi^epsilon_{\bar{a}}
        pi_eps = np.ones((S, A)) * epsilon / A
        for s in range(S):
            pi_eps[s, policy[s]] += (1 - epsilon)

        # Compute stationary distribution d^epsilon_{\bar{a}}
        # This is the stationary distribution of the Markov chain under pi_eps
        # P^pi(s'|s) = sum_a pi(s|a) P(s'|s,a)
        P_pi = np.zeros((S, S))
        for s in range(S):
            for s_next in range(S):
                for a in range(A):
                    P_pi[s, s_next] += pi_eps[s, a] * P[s, a, s_next]

        # Solve for stationary distribution: d = d @ P_pi, d ones = 1
        # This is the left eigenvector of P_pi with eigenvalue 1
        eigenvalues, eigenvectors = np.linalg.eig(P_pi.T)
        idx = np.argmin(np.abs(eigenvalues - 1.0))
        d_eps = np.real(eigenvectors[:, idx])
        d_eps = d_eps / d_eps.sum()  # Normalize
        d_eps = np.real(d_eps)  # Ensure real

        # Build diagonal matrix D^epsilon_{\bar{a}}
        # D^epsilon_{\bar{a}} is diagonal with (s,a)-th entry = d^epsilon_{\bar{a}}(s) * pi^epsilon_{\bar{a}}(a|s)
        D_eps = np.zeros((S * A, S * A))
        for s in range(S):
            for a in range(A):
                idx_sa = s * A + a
                D_eps[idx_sa, idx_sa] = d_eps[s] * pi_eps[s, a]

        # Compute expected reward r(s,a) = sum_{s'} P(s'|s,a) * r(s,a,s')
        r_expected = np.zeros(S * A)
        for s in range(S):
            for a in range(A):
                idx_sa = s * A + a
                r_expected[idx_sa] = np.sum(P[s, a, :] * r[s, a, :])

        # Compute b_{\bar{a}} = \Phi^T D^epsilon_{\bar{a}} r
        b_a = Phi.T @ D_eps @ r_expected

        # For Q-learning, we use epsilon' = 0 (greedy at next state)
        # P^{\epsilon'}_{\bar{a}}((s,a), (s',a')) = P(s'|s,a) * pi^{\epsilon'}_{\bar{a}}(a'|s')
        # For epsilon' = 0, this is just P(s'|s,a) * 1[a' = \bar{a}(s')]
        P_eps_prime = np.zeros((S * A, S * A))
        for s in range(S):
            for a in range(A):
                for s_next in range(S):
                    # For Q-learning (epsilon' = 0), next action is greedy
                    a_next = policy[s_next]
                    idx_sa = s * A + a
                    idx_next = s_next * A + a_next
                    P_eps_prime[idx_sa, idx_next] = P[s, a, s_next]

        # Compute A_{\bar{a}} = \Phi^T D^epsilon_{\bar{a}} (I - \gamma P^{\epsilon'}_{\bar{a}}) \Phi
        A_a = Phi.T @ D_eps @ (np.eye(S * A) - gamma * P_eps_prime) @ Phi

        results[tuple(policy)] = (b_a, A_a)

    return results


def compute_landmark(b_a, A_a):
    """
    Compute the landmark (equilibrium point) for a greedy region.
    The landmark is A_a^{-1} b_a, which is the equilibrium of θ̇ = b_a - A_a θ.

    Args:
        b_a: Vector b_{\bar{a}}
        A_a: Matrix A_{\bar{a}}

    Returns:
        np.ndarray: The landmark point θ* = A_a^{-1} b_a
    """
    try:
        return np.linalg.solve(A_a, b_a)
    except np.linalg.LinAlgError:
        return None