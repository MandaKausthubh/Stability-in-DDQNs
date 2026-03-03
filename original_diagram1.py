import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm

from environments.DiscreteMDPs import DiscreteMDP
from utils.ReplayBuffer import ReplayBuffer
from models.model import FeatureExtractor
from Experiments import DQN_GeneralFA


def create_random_mdp(S, A, seed):
    rng = np.random.default_rng(seed)
    P = rng.random((S, A, S))
    P /= P.sum(axis=2, keepdims=True)
    r = rng.random((S, A, S))
    return P, r


def run_single_experiment(seed, Phi=None, env=None):
    S = 10
    A = 10
    gamma = 0.99
    representation_dim = 5
    n_iterations = 3000

    np.random.seed(seed)
    torch.manual_seed(seed)

    print(f"Running experiment with seed {seed}...")

    if env is None:
        P, r = create_random_mdp(S, A, seed)
        env = DiscreteMDP(S, A, P, r)

    model = FeatureExtractor(
        input_dim=S,
        hidden_dims=[32],
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
        delta=0.1
    )

    if Phi is None:
        agent.pretrain(epochs=10000)
        agent.compute_phi_matrix()
        Phi = agent.Phi

    # Optimal reference
    Q_star = env.compute_optimal_Q(gamma)
    V_star = np.max(Q_star, axis=1)

    # ---- Initial policy gap ----
    Q_init = agent.compute_Q()
    pi_init = np.argmax(Q_init, axis=1)
    V_init = agent.compute_value_of_policy(pi_init, gamma)

    init_gap = np.max(np.abs(V_star - V_init))

    # ---- Train ----
    agent.learn(n_iterations=n_iterations, log_every=100000)

    # ---- Final policy gap ----
    Q_final = agent.compute_Q()
    pi_final = np.argmax(Q_final, axis=1)
    V_final = agent.compute_value_of_policy(pi_final, gamma)

    final_gap = np.max(np.abs(V_star - V_final))

    return init_gap, final_gap, Phi, env


def main():
    n_runs = 20
    init_gaps = []
    final_gaps = []

    Phi, env = None, None

    for seed in (range(n_runs)):
        print(f"\n=== Run {seed+1}/{n_runs} ===")
        g0, gT, Phi, env = run_single_experiment(seed, Phi, env)
        init_gaps.append(g0)
        final_gaps.append(gT)

    # Scatter plot
    plt.figure(figsize=(6,6))
    plt.scatter(init_gaps, final_gaps, alpha=0.7)
    max_val = max(init_gaps + final_gaps)

    plt.plot([0, max_val], [0, max_val], 'r--')
    plt.xlabel("Initial Value Gap")
    plt.ylabel("Final Value Gap")
    plt.title("DQN: Initial vs Final Value Suboptimality")
    plt.grid(True)
    plt.show()
    plt.savefig("dqn_value_gap_scatter.png")


if __name__ == "__main__":
    main()
