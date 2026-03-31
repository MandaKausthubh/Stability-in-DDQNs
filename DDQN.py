import numpy as np
from Experiments import DQN_GeneralFA


class DDQN_GeneralFA(DQN_GeneralFA):
    def initialise_theta(self):
        # NIPS 2010 Double Q-learning maintains TWO independent sets of weights
        self.theta_A = np.random.rand(self.representation_dim)
        self.theta_B = np.random.rand(self.representation_dim)
        
        # Maintain references for the parent class' logging/plotting compatibility
        self.theta = self.theta_A
        self.theta_target = self.theta_B
        self.theta_history = [self.theta.copy()]

    def compute_Q(self, theta=None):
        if theta is None:
            # The behavioral policy / evaluation Q-values are often the average of the two
            theta = (self.theta_A + self.theta_B) / 2.0
            
        return self.Phi.reshape(
            self.num_states,
            self.num_actions,
            self.representation_dim
        ) @ theta

    def _get_q_vals(self, theta):
        return self.Phi.reshape(
            self.num_states, self.num_actions, self.representation_dim
        ) @ theta

    def _optimizer_step(
        self,
        iter_idx: int,
        sample_from_stationary: bool = True,
    ):
        # 1. Choose one estimator to update at random
        update_A = self.rng.random() < 0.5
        
        if update_A:
            theta_update = self.theta_A
            theta_eval = self.theta_B
        else:
            theta_update = self.theta_B
            theta_eval = self.theta_A
            
        # Temporarily adapt self.theta so the parent's sample_trajectories uses the update network for the epsilon-greedy behavior
        self.theta = theta_update
        self.theta_target = theta_eval
        
        # 2. Sample 1 batch of transitions
        # We ignore the parent's next_a because it's sampled eps-greedy and we want a pure argmax
        s, a, r, next_s, _ = self.sample_trajectories(sample_from_stationary)
        
        # 3. Double Q-learning target
        # Sample the argmax from one (theta_update)
        Q_next_update = self._get_q_vals(theta_update)[next_s]  # Shape: (B, A)
        next_a = np.argmax(Q_next_update, axis=1)
        
        # Evaluate it using the other (theta_eval)
        sa_index = (s * self.num_actions) + a
        next_sa_index = (next_s * self.num_actions) + next_a
        
        phi_s = self.Phi[sa_index]
        phi_next = self.Phi[next_sa_index]
        
        # TD Target = r + gamma * Q_eval(s', argmax_a Q_update(s', a))
        td_error = r + self.gamma * (phi_next @ theta_eval) - (phi_s @ theta_update)
        grad = (td_error[:, None] * phi_s).mean(axis=0)
        
        # 4. Update the chosen estimator
        if update_A:
            self.theta_A += self.alpha(iter_idx) * grad
            self.theta = self.theta_A
            self.theta_target = self.theta_B
        else:
            self.theta_B += self.alpha(iter_idx) * grad
            self.theta = self.theta_B
            self.theta_target = self.theta_A
            
        return self.theta_A, self.theta_B