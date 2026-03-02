import torch
import numpy as np
import copy

from .models.model import FeatureExtractor
from .utils.ReplayBuffer import ReplayBuffer
from .utils.Sampling import get_abar, get_stationary_dist
from .environments.DiscreteMDPs import DiscreteMDP



class DQN_GeneralFA:
    def __init__(
        self,
        num_states: int,
        num_actions: int,
        model: FeatureExtractor,
        representation_dim: int,
        env: DiscreteMDP,
        replay_buffer: ReplayBuffer,
        eps: float = 0.1,
        gamma=0.99,
        batch_size=64,
        target_update_freq=100,
        delta=0.1,
        seed=10,
        pretrain_epochs=1000
    ):
        self.num_states = num_states
        self.num_actions = num_actions
        self.model = model
        self.representation_dim = representation_dim

        assert self.representation_dim == model.layers[-1].out_features, \
            "Representation dimension must match the output dimension of the model's last layer."

        assert self.representation_dim * self.num_actions == model.layers[-1].out_features, \
            "The output dimension of the model must be equal to representation_dim * num_actions."

        self.target_model = copy.deepcopy(model)
        self.target_model.load_state_dict(self.model.state_dict())

        self.env = env
        self.eps = eps
        self.gamma = gamma
        self.delta = delta
        self.replay_buffer = replay_buffer

        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.rng = np.random.default_rng(seed)
        self.mu = np.ones(num_states) / num_states  # uniform distribution over states
        self.l = self.replay_buffer.max_size
        self.theta = None
        self.initialise_theta()
        self.pretrain(pretrain_epochs)
        self.Phi : np.ndarray
        self.compute_phi_matrix()


    def pretrain(self, epochs=1000):
        Q_star = self.env.compute_optimal_Q()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        loss_fn = torch.nn.MSELoss()

        S, A = self.num_states, self.num_actions
        theta_temp = torch.rand(self.representation_dim, requires_grad=True)

        for _ in range(epochs):
            one_hot_states = torch.eye(S)
            features = self.model(one_hot_states)  # Shape: (S, representation_dim * A)
            features = features.view(S, A, self.representation_dim)  # Reshape to (S, A, representation_dim)
            Q_pred = torch.einsum('sak,k->sa', features, theta_temp)  # Shape: (S, A)
            loss = loss_fn(Q_pred, torch.tensor(Q_star, dtype=torch.float32))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        for p in self.model.parameters():
            p.requires_grad = False


    def reset_theta(self):
        self.initialise_theta()

    def compute_phi_matrix(self):
        Phi = []
        with torch.no_grad():
            for s in range(self.num_states):
                state_vec = torch.zeros(self.num_states)
                state_vec[s] = 1.0

                features = self.model(state_vec.unsqueeze(0)).squeeze(0)
                features = features.reshape(self.num_actions, self.representation_dim)
                for a in range(self.num_actions):
                    Phi.append(features[a].numpy())
        self.Phi = np.stack(Phi)


    def initialise_theta(self):
        # Randomly initialize theta
        self.theta = np.random.rand(self.representation_dim)
        self.theta_target = copy.deepcopy(self.theta)
        self.theta_history = [self.theta.copy()]


    def select_action(self, states, theta=None):
        if theta is None:
            theta = self.theta

        if self.rng.random() < self.eps:
            return self.rng.integers(self.num_actions, size=states.shape[0])

        else:
            if self.Phi is None:
                self.compute_phi_matrix()
            Q_vals = self.Phi.reshape(   # type: ignore
                self.num_states, self.num_actions, self.representation_dim
            ) @ theta
            # Dimensions: (num_states, num_actions, representation_dim) @ (representation_dim,) -> (num_states, num_actions)
            # Argmax over actions for each state
            return np.argmax(Q_vals[states], axis=1)


    def alpha(self, iter_idx):
        # Step size schedule, can be tuned as needed
        return 0.1 / (1 + iter_idx)

    def tau(self, iter_idx):
        # Target update schedule, can be tuned as needed
        return 0.1 / np.sqrt(1 + iter_idx)


    def generate_trajectory_sampes(
        self,
        iter_idx: int,
        sample_from_stationary: bool = False,
    ):
        s, a, r, next_s, next_a = self.sample_trajectories(sample_from_stationary)

        s = np.asarray(s)
        a = np.asarray(a)
        r = np.asarray(r)
        next_s = np.asarray(next_s)
        next_a = np.asarray(next_a)

        sa_index = (s*self.num_actions) + a
        next_sa_index = (next_s*self.num_actions) + next_a

        phi_s = self.Phi[sa_index]
        phi_next = self.Phi[next_sa_index]

        td_error = r + self.gamma * (phi_next @ self.theta_target) - (phi_s @ self.theta)
        grad = (td_error[:, None] * phi_s).mean(axis=0)
        self.theta += self.alpha(iter_idx) * grad

        if self.rng.random() < self.delta:
            self.theta_target += self.tau(iter_idx) * (
                self.theta - self.theta_target
            )
        self.replay_buffer.push(s, a, r, next_s, self.theta.copy())

    def build_eps_policy(self, abar):
        """
        Returns policy matrix of shape (S, A)
        """
        S, A = self.num_states, self.num_actions
        pi = np.ones((S, A)) * (self.eps / A)

        for s in range(S):
            pi[s, abar[s]] += (1 - self.eps)

        return pi

    def build_transition_matrix(self, pi):
        """
        Returns P^pi of shape (S, S)
        """
        S, A = self.num_states, self.num_actions
        P_pi = np.zeros((S, S))

        for s in range(S):
            for a in range(A):
                P_pi[s] += pi[s, a] * self.env.P[s, a]

        return P_pi

    def sample_trajectories(self, sample_from_stationary):
        B = self.batch_size
        S, A = self.num_states, self.num_actions

        s_batch = []
        a_batch = []
        r_batch = []
        next_s_batch = []
        next_a_batch = []

        for _ in range(B):
            if sample_from_stationary:
                k = self.rng.integers(self.l)
                theta_k = self.replay_buffer.theta_buf[k]
            else:
                theta_k = self.theta
            abar = get_abar(S, A, self.Phi, theta_k)
            pi_eps = self.build_eps_policy(abar)
            P_pi = self.build_transition_matrix(pi_eps)
            d = get_stationary_dist(P_pi)

            s = self.rng.choice(S, p=d)
            a = self.rng.choice(A, p=pi_eps[s])
            next_s = self.rng.choice(S, p=self.env.P[s, a])

            abar_next = get_abar(S, A, self.Phi, self.theta_target)
            pi_eps_next = self.build_eps_policy(abar_next)
            next_a = self.rng.choice(A, p=pi_eps_next[next_s])

            r = self.env.r[s, a, next_s]

            s_batch.append(s)
            a_batch.append(a)
            r_batch.append(r)
            next_s_batch.append(next_s)
            next_a_batch.append(next_a)

        return (
            np.array(s_batch),
            np.array(a_batch),
            np.array(r_batch),
            np.array(next_s_batch),
            np.array(next_a_batch)
        )











