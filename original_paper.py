import numpy as np
import torch

from environments.DiscreteMDPs import DiscreteMDP
from utils.ReplayBuffer import ReplayBuffer
from models.model import FeatureExtractor
from Experiments import DQN_GeneralFA

def create_random_mdp(S, A, seed=0):
    rng = np.random.default_rng(seed)

    P = rng.random((S, A, S))
    P /= P.sum(axis=2, keepdims=True)

    r = rng.random((S, A, S))

    return P, r

def main():

    # ----------------------------
    # Experiment settings
    # ----------------------------
    S = 5
    A = 3
    gamma = 0.99
    n_iterations = 5000
    seed = 42

    np.random.seed(seed)
    torch.manual_seed(seed)

    # ----------------------------
    # Create random MDP
    # ----------------------------
    P, r = create_random_mdp(S, A, seed)

    env = DiscreteMDP(
        n_states=S,
        n_actions=A,
        P=P,
        r=r
    )

    # ----------------------------
    # Feature extractor
    # ----------------------------
    representation_dim = 4

    model = FeatureExtractor(
        input_dim=S,
        hidden_dims=[16, 16],
        output_dim=A * representation_dim
    )

    replay_buffer = ReplayBuffer(capacity=50, rep_dim=representation_dim)

    agent = DQN_GeneralFA(
        num_states=S,
        num_actions=A,
        model=model,
        representation_dim=representation_dim,
        env=env,
        replay_buffer=replay_buffer,
        eps=0.1,
        gamma=gamma,
        delta=0.1,
        log_dir="runs",
        run_name="linear_dqn_test"
    )

    # ----------------------------
    # Pretrain + Freeze Φ
    # ----------------------------
    agent.pretrain()
    agent.compute_phi_matrix()

    # ----------------------------
    # Compute optimal reference
    # ----------------------------
    Q_star = env.compute_optimal_Q(gamma)
    pi_star = np.argmax(Q_star, axis=1)

    print("Optimal policy:", pi_star)
    print("Optimal Q:\n", np.round(Q_star, 4))

    # ----------------------------
    # Train
    # ----------------------------
    agent.learn(
        n_iterations=n_iterations,
        log_every=200,
    )

    # ----------------------------
    # Final diagnostics
    # ----------------------------
    Q_final = agent.compute_Q()
    pi_final = np.argmax(Q_final, axis=1)

    print("\nFinal greedy policy:", pi_final)
    print("Final Q:\n", np.round(Q_final, 4))

    V_star = np.max(Q_star, axis=1)
    V_final = agent.compute_value_of_policy(pi_final, gamma)

    print("Final value gap:",
          np.max(np.abs(V_star - V_final)))


if __name__ == "__main__":
    main()
