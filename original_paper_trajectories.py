import numpy as np
import torch
import matplotlib.pyplot as plt

from tqdm import tqdm

from environments.DiscreteMDPs import DiscreteMDP
from utils.ReplayBuffer import ReplayBuffer
from models.model import FeatureExtractor
from Experiments import DQN_GeneralFA


# -------------------------------------------------
# 1. Small MDP (2-state, 2-action)
# -------------------------------------------------

def create_random_mdp(S, A, seed):
    rng = np.random.default_rng(seed)
    P = rng.random((S, A, S))
    P /= P.sum(axis=2, keepdims=True)
    r = rng.random((S, A, S))
    return P, r

def build_small_mdp():
    S = 10
    A = 10
    P, r = create_random_mdp(S, A, seed=0)
    return DiscreteMDP(S, A, P, r)


# -------------------------------------------------
# 2. Pretrain once to obtain shared Phi
# -------------------------------------------------

def pretrain_feature_map(env, representation_dim=2):

    S = env.n_states
    A = env.n_actions

    model = FeatureExtractor(
        input_dim=S,
        hidden_dims=[],
        output_dim=A * representation_dim
    )

    replay_buffer = ReplayBuffer(capacity=10, rep_dim=representation_dim)

    agent = DQN_GeneralFA(
        num_states=S,
        num_actions=A,
        model=model,
        representation_dim=representation_dim,
        env=env,
        replay_buffer=replay_buffer,
        eps=0.1,
        gamma=0.9,
        delta=0.1
    )

    print("Pretraining feature extractor once...")
    agent.pretrain(epochs=10000)

    agent.compute_phi_matrix()
    Phi = agent.Phi.copy()

    return Phi


# -------------------------------------------------
# 3. Run one downstream trajectory (θ-learning only)
# -------------------------------------------------

def run_single_trajectory(env, Phi, seed, n_iterations=5000):

    np.random.seed(seed)
    torch.manual_seed(seed)

    S = env.n_states
    A = env.n_actions
    representation_dim = Phi.shape[1]

    # Dummy model (won’t be trained)
    model = FeatureExtractor(
        input_dim=S,
        hidden_dims=[],
        output_dim=A * representation_dim
    )

    replay_buffer = ReplayBuffer(capacity=10, rep_dim=representation_dim)

    agent = DQN_GeneralFA(
        num_states=S,
        num_actions=A,
        model=model,
        representation_dim=representation_dim,
        env=env,
        replay_buffer=replay_buffer,
        eps=0.3,
        gamma=0.4,
        delta=0.05,
        seed=seed
    )

    # Inject shared Phi
    agent.Phi = Phi.copy()

    # Reinitialize theta (different per seed)
    agent.reset_theta()

    theta_history = []

    for n in tqdm(range(n_iterations), desc=f"Trajectory {seed+1}", ncols=80):
        agent._optimizer_step(n, sample_from_stationary=True)
        theta_history.append(agent.theta.copy())

    return np.array(theta_history)


# -------------------------------------------------
# 4. Main: multiple trajectories, one plot
# -------------------------------------------------

def main():

    env = build_small_mdp()

    # ---- Shared representation ----
    Phi = pretrain_feature_map(env, representation_dim=2)

    n_runs = 5
    n_iterations = 10000

    plt.figure(figsize=(6,6))

    for seed in range(n_runs):
        print(f"\nRunning trajectory {seed+1}/{n_runs}")
        traj = run_single_trajectory(env, Phi, seed*10, n_iterations)
        plt.plot(traj[:,0], traj[:,1], alpha=0.8)
        plt.scatter(traj[0,0], traj[0,1], color='black', s=30)

    plt.xlabel("theta[0]")
    plt.ylabel("theta[1]")
    plt.title("Multiple Linear-DQN Trajectories (Shared Φ)")
    plt.grid(True)
    plt.show()
    plt.savefig("multi_trajectory_plot.png")


if __name__ == "__main__":
    main()
