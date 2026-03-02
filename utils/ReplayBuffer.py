# Original paper: "Does DQN Learn?" by Aditya Gopalan et al. (https://arxiv.org/pdf/2205.13617)
import numpy as np

class ReplayBuffer:
    """
    A simple replay buffer for storing transitions (s, a, r, next_s).
    """
    def __init__(self, capacity, rep_dim, seed=10):
        self.seed = seed
        self.max_size = capacity
        self.rep_dim = rep_dim
        self.s_buf = np.empty(self.max_size, dtype=int)
        self.a_buf = np.empty(self.max_size, dtype=int)
        self.r_buf = np.empty(self.max_size, dtype=float)
        self.next_s_buf = np.empty(self.max_size, dtype=int)
        self.theta_buf = np.empty((self.max_size, self.rep_dim), dtype=float)
        self.ptr, self.size = 0, 0
        # NOTE: Only change from the original paper,
        # they did not set the seed for the replay buffer,
        # which can lead to non-deterministic behavior in the experiments.
        np.random.seed(self.seed)

    def push(self, s, a, r, next_s, theta):
        self.s_buf[self.ptr] = s
        self.a_buf[self.ptr] = a
        self.r_buf[self.ptr] = r
        self.next_s_buf[self.ptr] = next_s
        self.theta_buf[self.ptr] = theta
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size, idxs=None):
        idxs = idxs if idxs is not None else np.random.choice(self.size, batch_size, replace=False)
        return list(zip(
            self.s_buf[idxs],
            self.a_buf[idxs],
            self.r_buf[idxs],
            self.next_s_buf[idxs],
            self.theta_buf[idxs]
        ))

    def reset(self):
        self.ptr = 0
        self.size = 0

    def __len__(self):
        return self.size





