"""
Linear DQN Implementation based on "Does DQN Learn?" paper by Gopalan & Thoppe.

This implements the idealized Linear DQN algorithm from the paper, which uses:
- Linear function approximation: Q(s,a) = φ(s,a)^T θ
- ε-greedy exploration
- Idealized experience replay (sampling from stationary distributions of past policies)
- Target network (updated on a faster timescale)
"""
import numpy as np
from typing import Tuple, List, Optional
from tqdm import tqdm


class LinearDQN:
    """
    Linear DQN implementation following the paper's theoretical framework.

    The algorithm maintains:
    - θ_n: current parameter estimate
    - θ_n^-: target network parameter (updated on faster timescale)

    Update rule (from paper equations 2-4):
        θ_{n+1} = θ_n + α_n δ_n φ(s_n, a_n)
        δ_n = r(s_n, a_n, s'_n) + γ φ(s'_n, a'_n)^T θ_n^- - φ(s_n, a_n)^T θ_n
        θ_{n+1}^- = θ_n^- + τ_n (θ_n - θ_n^-) ζ_{n+1}  (target network update)
    """

    def __init__(
        self,
        num_states: int,
        num_actions: int,
        Phi: np.ndarray,
        env,
        gamma: float = 0.9,
        epsilon: float = 0.05,
        epsilon_prime: float = 0.0,  # 0 for Q-learning, epsilon for SARSA
        alpha_type: str = "log_n",
        delta: float = 0.1,  # Target network refresh rate
        replay_length: int = 10,  # ℓ: replay buffer length
        seed: int = 42
    ):
        """
        Initialize Linear DQN.

        Args:
            num_states: Number of states S
            num_actions: Number of actions A
            Phi: Feature matrix of shape (SA, d)
            env: Environment with P, r attributes
            gamma: Discount factor
            epsilon: Exploration parameter for behavior policy
            epsilon_prime: Action sampling parameter (0 for Q-learning, epsilon for SARSA)
            alpha_type: Step size schedule type
            delta: Target network refresh probability per step
            replay_length: Number of past parameter vectors stored
            seed: Random seed
        """
        self.S = num_states
        self.A = num_actions
        self.Phi = Phi  # Shape (SA, d)
        self.d = Phi.shape[1]  # Representation dimension
        self.env = env
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_prime = epsilon_prime
        self.delta = delta
        self.L = replay_length

        self.rng = np.random.default_rng(seed)

        # Initialize parameters
        self.theta = np.zeros(self.d)
        self.theta_target = np.zeros(self.d)

        # Replay buffer for past theta values
        self.theta_history = [self.theta.copy() for _ in range(self.L)]
        self.history_pointer = 0

        # Step counters
        self.n = 0

        # Step size schedule
        self.alpha_type = alpha_type

        # Precompute some quantities
        self._precompute_stationary_dists()

    def _precompute_stationary_dists(self):
        """Precompute stationary distributions for all possible epsilon-greedy policies."""
        from itertools import product

        self.stationary_dists = {}
        self.eps_policies = {}

        for policy in product(range(self.A), repeat=self.S):
            policy = np.array(policy)

            # Build epsilon-greedy policy
            pi_eps = np.ones((self.S, self.A)) * self.epsilon / self.A
            for s in range(self.S):
                pi_eps[s, policy[s]] += (1 - self.epsilon)

            self.eps_policies[policy.tobytes()] = pi_eps

            # Compute stationary distribution
            P_pi = np.zeros((self.S, self.S))
            for s in range(self.S):
                for s_next in range(self.S):
                    for a in range(self.A):
                        P_pi[s, s_next] += pi_eps[s, a] * self.env.P[s, a, s_next]

            # Compute stationary distribution
            eigenvalues, eigenvectors = np.linalg.eig(P_pi.T)
            idx = np.argmin(np.abs(eigenvalues - 1.0))
            d = np.real(eigenvectors[:, idx])
            d = d / d.sum()
            d = np.real(d)

            self.stationary_dists[policy.tobytes()] = d

    def get_eps_policy(self, theta: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get the epsilon-greedy policy and its stationary distribution for a given theta.

        Args:
            theta: Parameter vector

        Returns:
            pi_eps: Epsilon-greedy policy of shape (S, A)
            d_eps: Stationary distribution of shape (S,)
        """
        # Compute Q-values
        Q = self.Phi @ theta  # Shape (SA,)
        Q = Q.reshape(self.S, self.A)

        # Get greedy policy
        greedy_policy = np.argmax(Q, axis=1)

        # Build epsilon-greedy policy
        pi_eps = np.ones((self.S, self.A)) * self.epsilon / self.A
        for s in range(self.S):
            pi_eps[s, greedy_policy[s]] += (1 - self.epsilon)

        # Compute stationary distribution
        policy_key = greedy_policy.tobytes()
        if policy_key in self.stationary_dists:
            d_eps = self.stationary_dists[policy_key]
        else:
            # Recompute if not cached
            P_pi = np.zeros((self.S, self.S))
            for s in range(self.S):
                for s_next in range(self.S):
                    for a in range(self.A):
                        P_pi[s, s_next] += pi_eps[s, a] * self.env.P[s, a, s_next]
            eigenvalues, eigenvectors = np.linalg.eig(P_pi.T)
            idx = np.argmin(np.abs(eigenvalues - 1.0))
            d_eps = np.real(eigenvectors[:, idx])
            d_eps = d_eps / d_eps.sum()

        return pi_eps, d_eps

    def get_greedy_policy(self, theta: np.ndarray) -> np.ndarray:
        """Get greedy policy for given theta."""
        Q = self.Phi @ theta
        Q = Q.reshape(self.S, self.A)
        return np.argmax(Q, axis=1)

    def get_q_values(self, theta: np.ndarray) -> np.ndarray:
        """Compute Q-values for given theta."""
        Q = self.Phi @ theta
        return Q.reshape(self.S, self.A)

    def alpha(self, n: int) -> float:
        """
        Step size schedule α_n.
        Paper uses α_n = log(n) / (10n) or α_n = c/n for various constants.
        """
        if self.alpha_type == "log_n":
            # Paper mentions α_n = log(n) / (10n)
            return np.log(n + 2) / (10.0 * (n + 2))
        elif self.alpha_type == "1_over_n":
            # α_n = c/n
            return 1.0 / (n + 1)
        elif self.alpha_type == "figure2":
            # Figure 2 uses α_n = 2/n
            return 2.0 / (n + 1)
        else:
            return 0.01

    def tau(self, n: int) -> float:
        """
        Target network update rate τ_n.
        Paper uses τ_n = 0.1 / sqrt(1 + n) or similar.
        """
        return 0.1 / np.sqrt(1 + n)

    def sample_trajectory(
        self,
        theta: np.ndarray,
        theta_target: np.ndarray,
        sample_from_buffer: bool = True
    ) -> Tuple[int, int, float, int, int]:
        """
        Sample a single transition (s, a, r, s', a') as per the paper's idealized experience replay.

        The paper samples:
        - k uniformly from {0, ..., ℓ} (or according to buffer distribution)
        - s from stationary distribution d^ε_{π_{n-k}}
        - a from π^ε_{n-k}(·|s)
        - s' from P(·|s,a)
        - a' from π^{ε'}_{n}(·|s') for Q-learning (ε' = 0) or SARSA (ε' = ε)

        Args:
            theta: Current parameter θ_n
            theta_target: Target network parameter θ_n^-
            sample_from_buffer: Whether to sample from past policies

        Returns:
            s: State
            a: Action
            r: Reward
            s_next: Next state
            a_next: Next action (sampled from ε'-greedy policy)
        """
        # Sample index k for experience replay
        if sample_from_buffer and self.L > 0:
            k = self.rng.integers(self.L)
            theta_k = self.theta_history[k]
        else:
            k = 0
            theta_k = theta

        # Get policy and stationary distribution for theta_k
        pi_eps, d_eps = self.get_eps_policy(theta_k)

        # Sample s from stationary distribution
        s = self.rng.choice(self.S, p=d_eps)

        # Sample a from epsilon-greedy policy
        a = self.rng.choice(self.A, p=pi_eps[s])

        # Sample s' from transition dynamics
        s_next = self.rng.choice(self.S, p=self.env.P[s, a])

        # Get reward r(s,a,s')
        r = self.env.r[s, a, s_next]

        # Sample a' from epsilon'-greedy policy at s' using target network
        Q_target = self.Phi @ theta_target
        Q_target = Q_target.reshape(self.S, self.A)
        greedy_a_next = np.argmax(Q_target[s_next])

        if self.epsilon_prime > 0:
            # SARSA: sample from epsilon-greedy
            probs = np.ones(self.A) * self.epsilon_prime / self.A
            probs[greedy_a_next] += (1 - self.epsilon_prime)
            a_next = self.rng.choice(self.A, p=probs)
        else:
            # Q-learning: greedy (epsilon' = 0)
            a_next = greedy_a_next

        return s, a, r, s_next, a_next

    def sample_batch(
        self,
        batch_size: int,
        theta: np.ndarray,
        theta_target: np.ndarray,
        sample_from_buffer: bool = True
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Sample a batch of transitions."""
        s_batch = np.zeros(batch_size, dtype=int)
        a_batch = np.zeros(batch_size, dtype=int)
        r_batch = np.zeros(batch_size)
        s_next_batch = np.zeros(batch_size, dtype=int)
        a_next_batch = np.zeros(batch_size, dtype=int)

        for i in range(batch_size):
            s, a, r, s_next, a_next = self.sample_trajectory(
                theta, theta_target, sample_from_buffer
            )
            s_batch[i] = s
            a_batch[i] = a
            r_batch[i] = r
            s_next_batch[i] = s_next
            a_next_batch[i] = a_next

        return s_batch, a_batch, r_batch, s_next_batch, a_next_batch

    def optimizer_step(
        self,
        batch_size: int = 8,
        sample_from_buffer: bool = True
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Perform one optimization step following the paper's update rule.

        Returns:
            theta: Updated parameter
            theta_target: Updated target parameter
        """
        # Sample batch
        s, a, r, s_next, a_next = self.sample_batch(
            batch_size, self.theta, self.theta_target, sample_from_buffer
        )

        # Get feature vectors
        sa_indices = s * self.A + a
        sa_next_indices = s_next * self.A + a_next

        phi_s = self.Phi[sa_indices]  # Shape (batch, d)
        phi_next = self.Phi[sa_next_indices]  # Shape (batch, d)

        # Compute TD error
        td_error = r + self.gamma * (phi_next @ self.theta_target) - (phi_s @ self.theta)

        # Compute gradient and update
        grad = (td_error[:, np.newaxis] * phi_s).mean(axis=0)

        alpha_n = self.alpha(self.n)
        self.theta = self.theta + alpha_n * grad

        # Target network update (with probability delta)
        if self.rng.random() < self.delta:
            tau_n = self.tau(self.n)
            self.theta_target = self.theta_target + tau_n * (self.theta - self.theta_target)

        # Update history
        self.theta_history[self.history_pointer] = self.theta.copy()
        self.history_pointer = (self.history_pointer + 1) % self.L

        self.n += 1

        return self.theta.copy(), self.theta_target.copy()

    def reset(self, theta_init: Optional[np.ndarray] = None, seed: Optional[int] = None):
        """Reset the algorithm to initial state."""
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        if theta_init is not None:
            self.theta = theta_init.copy()
        else:
            self.theta = np.zeros(self.d)

        self.theta_target = self.theta.copy()
        self.theta_history = [self.theta.copy() for _ in range(self.L)]
        self.history_pointer = 0
        self.n = 0

    def run(
        self,
        n_iterations: int,
        batch_size: int = 8,
        sample_from_buffer: bool = True,
        verbose: bool = False,
        log_interval: int = 100
    ) -> dict:
        """
        Run Linear DQN for specified number of iterations.

        Args:
            n_iterations: Number of iterations to run
            batch_size: Batch size for each update
            sample_from_buffer: Whether to sample from past policies
            verbose: Whether to show progress bar
            log_interval: How often to log progress

        Returns:
            Dictionary with trajectory history
        """
        theta_trajectory = []
        theta_target_trajectory = []

        iterator = range(n_iterations)
        if verbose:
            iterator = tqdm(iterator, desc="Linear DQN")

        for _ in iterator:
            theta, theta_target = self.optimizer_step(batch_size, sample_from_buffer)
            theta_trajectory.append(theta.copy())
            theta_target_trajectory.append(theta_target.copy())

        return {
            'theta_trajectory': np.array(theta_trajectory),
            'theta_target_trajectory': np.array(theta_target_trajectory)
        }

    def compute_value_of_policy(self, policy: np.ndarray, gamma: Optional[float] = None) -> np.ndarray:
        """
        Compute V^π exactly by solving the Bellman linear system.

        Args:
            policy: Deterministic policy of shape (S,)
            gamma: Discount factor (uses self.gamma if None)

        Returns:
            V^π: Value function of shape (S,)
        """
        if gamma is None:
            gamma = self.gamma

        # Build P^π and r^π
        P_pi = np.zeros((self.S, self.S))
        r_pi = np.zeros(self.S)

        for s in range(self.S):
            a = policy[s]
            P_pi[s] = self.env.P[s, a]
            r_pi[s] = np.sum(self.env.P[s, a] * self.env.r[s, a])

        # Solve (I - γP^π)V = r^π
        V = np.linalg.solve(np.eye(self.S) - gamma * P_pi, r_pi)
        return V

    def compute_optimal_value(self, gamma: Optional[float] = None) -> np.ndarray:
        """Compute V* using value iteration."""
        if gamma is None:
            gamma = self.gamma

        Q = self.env.compute_optimal_Q(gamma)
        V = np.max(Q, axis=1)
        return V