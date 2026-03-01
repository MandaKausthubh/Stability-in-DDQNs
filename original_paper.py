


# Original paper: "Does DQN Learn?" by Aditya Gopalan et al. (https://arxiv.org/pdf/2205.13617)

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm




class ReplayBuffer:

    def __init__(self, capacity, seed=10):
        self.seed = seed
        self.max_size = capacity
        self.s_buf = np.empty(self.max_size, dtype=int)
        self.a_buf = np.empty(self.max_size, dtype=int)
        self.r_buf = np.empty(self.max_size, dtype=float)
        self.next_s_buf = np.empty(self.max_size, dtype=int)
        self.ptr, self.size = 0, 0
        # NOTE: Only change from the original paper,
        # they did not set the seed for the replay buffer,
        # which can lead to non-deterministic behavior in the experiments.
        np.random.seed(self.seed)


    def push(self, s, a, r, next_s):
        self.s_buf[self.ptr] = s
        self.a_buf[self.ptr] = a
        self.r_buf[self.ptr] = r
        self.next_s_buf[self.ptr] = next_s
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)


    def sample(self, batch_size):
        idxs = np.random.choice(self.size, batch_size, replace=False)
        return list(zip(
            self.s_buf[idxs],
            self.a_buf[idxs],
            self.r_buf[idxs],
            self.next_s_buf[idxs]
        ))

    def reset(self):
        self.ptr = 0
        self.size = 0

    def __len__(self):
        return self.size



class DQNAgent:
    """
    DQNs with a generale function approximation architecture, as described in the original Does DQN Learn? paper
    """

    def __init__(
        self,
        env, # A discrete MDP environment with a finite state and action space
        S, # Number of states
        A, # Number of actions
        Q_net, # A neural network that takes in a state (one-hot) and outputs a vector of action values
        optimizer,
        epsilon,
        gamma,
        replay_buf,
        batch_size,
        target_update_period
    ) -> None:
        self.S = S
        self.A = A
        self.env = env
        self.Q_net = Q_net
        self.Q_net_target = Q_net
        self.epsilon = epsilon
        self.gamma = gamma
        self.memory = replay_buf
        self.batch_size = batch_size
        self.target_update_period = target_update_period
        self.steps_done = 0
        self.optimizer = optimizer

    def select_action(self, state, greedy=False):
        if greedy:
            eps_threshold = 0.0
        else:
            eps_threshold = self.epsilon

        self.steps_done += 1
        sample = np.random.rand()
        if sample > eps_threshold:
            with torch.no_grad():
                return self.Q_net(state).argmax().item()
        else:
            return np.random.randint(self.A)


    def _optimizer_model(self, iter_idx, sample_from_stationary=False):
        if len(self.memory) < self.batch_size:
            return None

        transitions = self.memory.sample(self.batch_size)
        s_batch, a_batch, r_batch, next_s_batch = [], [], [], []
        for transition in transitions:
            if sample_from_stationary:
                # TODO: Implement this sampling utils method
                s, a, r, next_s = transition
            else: s, a, r, next_s = transition
            s_batch.append(s)
            a_batch.append(a)  
            r_batch.append(r)
            next_s_batch.append(next_s)

        s = torch.tensor(s_batch, dtype=torch.float32)
        a = torch.tensor(a_batch, dtype=torch.int64)
        r = torch.tensor(r_batch, dtype=torch.float32)
        next_s = torch.tensor(next_s_batch, dtype=torch.float32)

        with torch.no_grad():
            target_q_values = r + self.gamma * self.Q_net_target(next_s).max(dim=1)[0]

        q_values = self.Q_net(s).gather(1, a.unsqueeze(1)).squeeze()
        loss = torch.nn.functional.mse_loss(q_values, target_q_values)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


    def learn(
        self,
        step_sizes,
        seed=10,
        sample_from_stationary=False,
        learn_after_iter=0,
        verbose=True
    ):
        self.step_sizes = step_sizes
        self.num_iters = len(step_sizes)
        self.rng = np.random.default_rng(seed)
        self.memory.reset(seed=seed)
        self.Q_net.reset()
        self.trajectory = []

        self.state = self.env.reset(seed=seed)
        iter_obj = tqdm(range(learn_after_iter + self.num_iters), desc="Training DQN", disable=not verbose)

        for iter in iter_obj:
            self.trajectory.append(self.Q_net.trainable_parameters())
            action = self.select_action(self.state)
            next_state, reward, done, _ = self.env.step(action)
            self.memory.push(self.state, action, reward, next_state)
            self.state = next_state
            if iter >= learn_after_iter:
                self._optimizer_model(iter, sample_from_stationary=sample_from_stationary)
                if iter % self.target_update_period == 0:
                    self.Q_net_target.load_state_dict(self.Q_net.state_dict())
            if done:
                self.state = self.env.reset(seed=seed)
        self.env.close()
        return self.trajectory



class QNet(nn.Module):

    def __init__(
        self, in_dim, out_dim,
        hidden_layers=[],
        nonlinearity=F.relu,
        bias=True
    ):
        """
        in_dim: state dim,
        out_dim: action_dim
        """
        super().__init__()
        self.layers = nn.ModuleList()
        self.nonlinearity = nonlinearity
        self.bias = bias

        prev_dim = in_dim
        for h_dim in hidden_layers:
            self.layers.append(nn.Linear(prev_dim, h_dim, bias=self.bias))
            prev_dim = h_dim

        if hidden_layers:
            self.output = nn.Linear(hidden_layers[-1], out_dim, bias=self.bias)
        else:
            self.output = nn.Linear(in_dim, out_dim, bias=self.bias)


    def reset(self):
        for layer in self.layers:
            layer.reset_parameters()     # type: ignore
        self.output.reset_parameters()




