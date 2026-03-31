#!/usr/bin/env python3
"""
Reproduce the experiments from "Does DQN Learn?" paper.

This script reproduces:
1. Figure 2: Three different trajectories on a 2-state, 2-action MDP
   - Convergence to optimal policy (blue)
   - Convergence to sub-optimal policy (green)
   - Policy oscillation (red - sliding mode attractor)

2. Figure 3b: MDP where linear DQN always converges to the worst policy

3. Figure 1: Scatterplot of initial vs final value suboptimality on random MDPs
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
import os
from typing import List, Tuple
from tqdm import tqdm

from environments.DiscreteMDPs import DiscreteMDP
from environments.MDPConstructs import create_figure2_mdp, create_figure3b_mdp, compute_babar, compute_landmark
from LinearDQN import LinearDQN


def run_trajectory(
    mdp,
    Phi,
    gamma,
    epsilon,
    theta_init,
    n_iterations=5000,
    batch_size=8,
    delta=0.1,  # Target network refresh rate
    replay_length=10,
    alpha_type="1_over_n",
    seed=None
):
    """
    Run a single trajectory of Linear DQN.

    Args:
        mdp: Environment
        Phi: Feature matrix
        gamma: Discount factor
        epsilon: Exploration parameter
        theta_init: Initial parameter
        n_iterations: Number of iterations
        batch_size: Batch size
        delta: Target network refresh probability
        replay_length: Replay buffer length
        alpha_type: Step size schedule type
        seed: Random seed

    Returns:
        Dictionary with results
    """
    S = mdp.n_states
    A = mdp.n_actions

    agent = LinearDQN(
        num_states=S,
        num_actions=A,
        Phi=Phi,
        env=mdp,
        gamma=gamma,
        epsilon=epsilon,
        epsilon_prime=0.0,  # Q-learning
        alpha_type=alpha_type,
        delta=delta,
        replay_length=replay_length,
        seed=seed
    )
    agent.reset(theta_init)

    results = agent.run(
        n_iterations=n_iterations,
        batch_size=batch_size,
        sample_from_buffer=True,
        verbose=False
    )

    return results


def plot_trajectory_with_regions(
    ax,
    theta_trajectory,
    Phi,
    mdp,
    gamma,
    epsilon,
    landmarks=None,
    title="",
    color='blue',
    show_regions=True
):
    """
    Plot trajectory with greedy region boundaries and landmarks.

    Args:
        ax: Matplotlib axis
        theta_trajectory: Array of shape (n_iterations, d)
        Phi: Feature matrix
        mdp: Environment
        gamma: Discount factor
        epsilon: Exploration parameter
        landmarks: List of landmark points
        title: Plot title
        color: Trajectory color
        show_regions: Whether to show greedy regions
    """
    S = mdp.n_states
    A = mdp.n_actions

    # Compute greedy region boundaries
    # For 2D, we have 4 policies: (a1,a1), (a1,a2), (a2,a1), (a2,a2)
    # Regions are defined by: theta such that argmax_a phi(s,a)^T theta is the policy

    if show_regions and Phi.shape[1] == 2:
        # Define policies
        policies = [(0, 0), (0, 1), (1, 0), (1, 1)]
        colors_regions = ['#a8e6cf', '#ffd3b6', '#dcedc1', '#ffaaa5']

        # For each policy, find the region boundaries
        for idx, policy in enumerate(policies):
            # Region R_policy: theta such that for all s, argmax_a phi(s,a)^T theta = policy[s]
            # This is a polyhedral cone
            # We'll just shade approximate regions by sampling
            pass

    # Plot trajectory
    ax.plot(theta_trajectory[:, 0], theta_trajectory[:, 1], color=color, alpha=0.7, linewidth=1)
    ax.scatter(theta_trajectory[0, 0], theta_trajectory[0, 1], color=color, s=50, marker='o', label='Start', zorder=5)
    ax.scatter(theta_trajectory[-1, 0], theta_trajectory[-1, 1], color=color, s=100, marker='*', label='End', zorder=5)

    # Plot landmarks if provided
    if landmarks is not None:
        for i, (lm, pol) in enumerate(landmarks):
            ax.scatter(lm[0], lm[1], color='black', s=100, marker='D', zorder=10)
            ax.annotate(f'$\\bar{{a}}_{i+1}$', (lm[0], lm[1]), fontsize=10, ha='center', va='bottom')

    ax.set_xlabel(r'$\theta_1$')
    ax.set_ylabel(r'$\theta_2$')
    ax.set_title(title)
    ax.grid(True, alpha=0.3)


def figure2_experiment():
    """
    Reproduce Figure 2 from the paper showing three different trajectories.
    """
    print("Running Figure 2 experiment...")
    mdp, Phi, gamma, epsilon = create_figure2_mdp()
    S, A = mdp.n_states, mdp.n_actions

    # Compute landmarks (equilibrium points for each greedy region)
    print("Computing landmarks...")
    results = compute_babar(mdp.P, mdp.r, Phi, epsilon, gamma, S, A)
    landmarks = []
    for policy, (b_a, A_a) in results.items():
        landmark = compute_landmark(b_a, A_a)
        if landmark is not None:
            landmarks.append((landmark, policy))
            print(f"  Policy {policy}: landmark = {landmark}")

    # Compute optimal Q* and its parameter theta*
    Q_star = mdp.compute_optimal_Q(gamma)
    # Find theta* such that Phi theta* approximates Q*
    # Since Phi has rank 2 and Q* is 4-dimensional, we find least squares solution
    theta_star = np.linalg.lstsq(Phi, Q_star.flatten(), rcond=None)[0]
    print(f"Optimal theta*: {theta_star}")

    # Initial theta: chosen so that initial policy is epsilon-greedy version of optimal
    # The optimal policy is the one that maximizes Q*
    optimal_policy = np.argmax(Q_star, axis=1)
    print(f"Optimal policy: {optimal_policy}")

    # Initial theta should be in the greedy region of optimal policy
    # Start from theta_star and perturb slightly
    theta_init_base = theta_star.copy()

    # Run multiple trajectories with different seeds
    n_iterations = 5000
    trajectories = []
    colors = ['blue', 'green', 'red']
    labels = ['Trajectory 1 (converges to optimal)', 'Trajectory 2 (sub-optimal)', 'Trajectory 3 (oscillation)']

    # Different initial perturbations to get different behaviors
    perturbations = [
        np.array([0.1, -0.1]),
        np.array([-0.1, 0.1]),
        np.array([0.05, 0.05])
    ]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for i, (perturb, color, label) in enumerate(zip(perturbations, colors, labels)):
        theta_init = theta_init_base + perturb
        print(f"\nRunning {label} with theta_init = {theta_init}")

        results = run_trajectory(
            mdp=mdp,
            Phi=Phi,
            gamma=gamma,
            epsilon=epsilon,
            theta_init=theta_init,
            n_iterations=n_iterations,
            batch_size=8,
            delta=0.125,
            replay_length=10,
            alpha_type="figure2",  # α_n = 2/n
            seed=42 + i
        )

        trajectories.append(results['theta_trajectory'])

        # Plot trajectory on left subplot (parameter space)
        ax = axes[0]
        traj = results['theta_trajectory']
        # Plot initial portion faded
        n_fade = min(500, len(traj))
        ax.plot(traj[:n_fade, 0], traj[:n_fade, 1], color=color, alpha=0.3, linewidth=1)
        # Plot rest solid
        ax.plot(traj[n_fade:, 0], traj[n_fade:, 1], color=color, alpha=0.9, linewidth=1.5)
        ax.scatter(traj[0, 0], traj[0, 1], color='black', s=100, marker='o', zorder=10)
        ax.scatter(traj[-1, 0], traj[-1, 1], color=color, s=150, marker='*', zorder=10)

        # Plot policy evolution on right subplot
        ax2 = axes[1]
        policies = [np.argmax(Phi @ traj[j]) for j in range(len(traj))]
        print(f"Shapes: traj = {traj.shape}, policies = {len(policies)}")
        # OUTPUT SHAPES: traj = (5000, 2), policies = 5000
        print(policies[:10])  # Print first 10 policies for debugging
        print(f"Unique policies: {set(policies)}")  # Check which policies are visited
        # policies are integers from 0 to 3 corresponding to (a1,a1), (a1,a2), (a2,a1), (a2,a2)
        # Convert policies to 0,1,2,3 for plotting
        policy_nums = policies  # Already in 0,1,2,3 format
        ax2.plot(policy_nums, color=color, alpha=0.7, linewidth=1)

    # Add landmarks to parameter space plot
    ax = axes[0]
    for lm, pol in landmarks:
        ax.scatter(lm[0], lm[1], color='black', s=150, marker='D', edgecolors='white', linewidths=2, zorder=15)
    # Mark optimal theta*
    ax.scatter(theta_star[0], theta_star[1], color='gold', s=200, marker='*', edgecolors='black', linewidths=2, zorder=20, label=r'$\theta^*$ (optimal)')

    ax.set_xlabel(r'$\theta_1$', fontsize=12)
    ax.set_ylabel(r'$\theta_2$', fontsize=12)
    ax.set_title('Parameter Space Trajectories', fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best')

    # Right subplot: Policy evolution
    ax2 = axes[1]
    ax2.set_xlabel('Iteration', fontsize=12)
    ax2.set_ylabel('Policy (encoded)', fontsize=12)
    ax2.set_title('Greedy Policy Evolution', fontsize=14)
    ax2.set_yticks([0, 1, 2, 3])
    ax2.set_yticklabels(['(a1,a1)', '(a1,a2)', '(a2,a1)', '(a2,a2)'])
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('figure2_reproduction.png', dpi=150, bbox_inches='tight')
    plt.savefig('figure2_reproduction.pdf', bbox_inches='tight')
    print("Saved figure2_reproduction.png and figure2_reproduction.pdf")
    plt.close()

    return trajectories, landmarks


def figure3b_experiment():
    """
    Reproduce Figure 3b: MDP where linear DQN converges to worst policy.
    """
    print("\nRunning Figure 3b experiment...")
    mdp, Phi, gamma, epsilon = create_figure3b_mdp()
    S, A = mdp.n_states, mdp.n_actions

    # Compute landmarks
    print("Computing landmarks...")
    results = compute_babar(mdp.P, mdp.r, Phi, epsilon, gamma, S, A)
    landmarks = []
    for policy, (b_a, A_a) in results.items():
        landmark = compute_landmark(b_a, A_a)
        if landmark is not None:
            landmarks.append((landmark, policy))
            print(f"  Policy {policy}: landmark = {landmark}")

    # Compute optimal Q* and value of each policy
    Q_star = mdp.compute_optimal_Q(gamma)
    print(f"\nOptimal Q*:\n{Q_star}")
    print(f"Optimal policy: {np.argmax(Q_star, axis=1)}")

    # Evaluate each policy's value
    print("\nPolicy values:")
    from itertools import product
    for policy in product(range(A), repeat=S):
        policy = np.array(policy)
        V = np.linalg.solve(
            np.eye(S) - gamma * mdp.P[:, policy[0], :] if S == 1 else
            np.array([[mdp.P[0, policy[0], 0], mdp.P[0, policy[0], 1]],
                      [mdp.P[1, policy[1], 0], mdp.P[1, policy[1], 1]]]),
            np.array([np.sum(mdp.P[0, policy[0]] * mdp.r[0, policy[0]]),
                      np.sum(mdp.P[1, policy[1]] * mdp.r[1, policy[1]])])
        )
        print(f"  Policy {tuple(policy)}: V = {V}, mean V = {V.mean():.4f}")

    # Run multiple trajectories from different starting points
    n_iterations = 3000
    trajectories = []

    # Start from multiple random initial points
    np.random.seed(42)
    n_starts = 5
    theta_inits = [np.random.randn(2) for _ in range(n_starts)]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for i, theta_init in enumerate(theta_inits):
        print(f"\nRunning from theta_init = {theta_init}")

        results = run_trajectory(
            mdp=mdp,
            Phi=Phi,
            gamma=gamma,
            epsilon=epsilon,
            theta_init=theta_init,
            n_iterations=n_iterations,
            batch_size=8,
            delta=0.125,
            replay_length=10,
            alpha_type="1_over_n",
            seed=42 + i
        )

        trajectories.append(results['theta_trajectory'])

        # Plot trajectory
        ax = axes[0]
        traj = results['theta_trajectory']
        color = plt.cm.tab10(i)
        ax.plot(traj[:, 0], traj[:, 1], color=color, alpha=0.7, linewidth=1)
        ax.scatter(traj[0, 0], traj[0, 1], color=color, s=50, marker='o', zorder=5)
        ax.scatter(traj[-1, 0], traj[-1, 1], color=color, s=100, marker='*', zorder=5)

        # Plot final policy
        ax2 = axes[1]
        final_theta = traj[-1]
        final_Q = Phi @ final_theta
        final_Q = final_Q.reshape(S, A)
        final_policy = np.argmax(final_Q, axis=1)
        print(f"  Final theta: {final_theta}")
        print(f"  Final policy: {final_policy}")

    # Add landmarks
    ax = axes[0]
    for lm, pol in landmarks:
        ax.scatter(lm[0], lm[1], color='black', s=150, marker='D', edgecolors='white', linewidths=2, zorder=15)

    ax.set_xlabel(r'$\theta_1$', fontsize=12)
    ax.set_ylabel(r'$\theta_2$', fontsize=12)
    ax.set_title('Parameter Space (Worst Policy MDP)', fontsize=14)
    ax.grid(True, alpha=0.3)

    axes[1].set_xlabel('Iteration', fontsize=12)
    axes[1].set_ylabel('Policy', fontsize=12)
    axes[1].set_title('Policy Evolution', fontsize=14)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('figure3b_reproduction.png', dpi=150, bbox_inches='tight')
    plt.savefig('figure3b_reproduction.pdf', bbox_inches='tight')
    print("Saved figure3b_reproduction.png and figure3b_reproduction.pdf")
    plt.close()

    return trajectories, landmarks


def figure1_experiment(n_mdps=100):
    """
    Reproduce Figure 1: Scatterplot of initial vs final value suboptimality.
    """
    print(f"\nRunning Figure 1 experiment with {n_mdps} random MDPs...")

    S, A = 10, 10  # 10 states, 10 actions
    gamma = 0.75
    n_iterations = 5000
    batch_size = 4
    epsilon = 0.2

    initial_gaps = []
    final_gaps = []

    for mdp_idx in tqdm(range(n_mdps), desc="Processing MDPs"):
        # Generate random MDP
        np.random.seed(mdp_idx * 1000)

        # Random transition probabilities (normalize rows)
        P = np.random.rand(S, A, S)
        P = P / P.sum(axis=2, keepdims=True)

        # Random rewards
        r = np.random.rand(S, A, S)

        # Create MDP
        rho = np.ones(S) / S
        mdp = DiscreteMDP(S, A, P, r, rho)

        # Compute optimal value
        Q_star = mdp.compute_optimal_Q(gamma)
        V_star = np.max(Q_star, axis=1)

        # Random feature matrix: Phi has shape (SA, d) where d is representation dimension
        # Paper uses hidden layer width 1, output width A=10
        # So representation dimension d = 1 * A = 10 (actually paper says hidden width 1)
        # Let's use d = 10 for reasonable approximation
        d = 10
        Phi = np.random.randn(S * A, d)

        # Normalize features
        Phi = Phi / np.linalg.norm(Phi, axis=0, keepdims=True)

        # Random initial theta
        theta_init = np.random.randn(d)

        # Initialize Linear DQN
        agent = LinearDQN(
            num_states=S,
            num_actions=A,
            Phi=Phi,
            env=mdp,
            gamma=gamma,
            epsilon=epsilon,
            epsilon_prime=0.0,
            alpha_type="log_n",
            delta=0.25,  # Target network update every 4 steps
            replay_length=100,
            seed=mdp_idx
        )
        agent.reset(theta_init)

        # Compute initial policy value
        Q_init = Phi @ theta_init
        Q_init = Q_init.reshape(S, A)
        policy_init = np.argmax(Q_init, axis=1)
        V_init = agent.compute_value_of_policy(policy_init, gamma)
        gap_init = np.max(np.abs(V_star - V_init))

        # Run DQN
        agent.run(n_iterations=n_iterations, batch_size=batch_size, sample_from_buffer=True)

        # Compute final policy value
        Q_final = Phi @ agent.theta
        Q_final = Q_final.reshape(S, A)
        policy_final = np.argmax(Q_final, axis=1)
        V_final = agent.compute_value_of_policy(policy_final, gamma)
        gap_final = np.max(np.abs(V_star - V_final))

        initial_gaps.append(gap_init)
        final_gaps.append(gap_final)

    # Create scatter plot
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(initial_gaps, final_gaps, alpha=0.5, s=30)

    # Plot diagonal
    max_val = max(max(initial_gaps), max(final_gaps))
    ax.plot([0, max_val], [0, max_val], 'r--', label='y = x (no improvement line)', linewidth=2)

    ax.set_xlabel('Initial Value Suboptimality', fontsize=12)
    ax.set_ylabel('Final Value Suboptimality', fontsize=12)
    ax.set_title('DQN Performance: Initial vs Final Policy Value Gap', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Add statistics
    worse_count = sum(1 for i, f in zip(initial_gaps, final_gaps) if f > i)
    ax.text(0.05, 0.95, f'{worse_count}/{n_mdps} ({100*worse_count/n_mdps:.1f}%) worse than initial',
            transform=ax.transAxes, fontsize=10, verticalalignment='top')

    plt.tight_layout()
    plt.savefig('figure1_reproduction.png', dpi=150, bbox_inches='tight')
    plt.savefig('figure1_reproduction.pdf', bbox_inches='tight')
    print("Saved figure1_reproduction.png and figure1_reproduction.pdf")
    plt.close()

    return initial_gaps, final_gaps


def main():
    """Run all experiments."""
    print("=" * 60)
    print("Reproducing experiments from 'Does DQN Learn?' paper")
    print("=" * 60)

    # Create output directory
    os.makedirs('results', exist_ok=True)
    os.chdir('results')

    # Run Figure 2 experiment
    print("\n" + "=" * 60)
    print("FIGURE 2: Three trajectories with different behaviors")
    print("=" * 60)
    trajectories_fig2, landmarks_fig2 = figure2_experiment()

    # Run Figure 3b experiment
    print("\n" + "=" * 60)
    print("FIGURE 3b: MDP where DQN converges to worst policy")
    print("=" * 60)
    trajectories_fig3b, landmarks_fig3b = figure3b_experiment()

    # Run Figure 1 experiment (more computationally intensive)
    print("\n" + "=" * 60)
    print("FIGURE 1: Random MDP scatterplot")
    print("=" * 60)
    # Use fewer MDPs for quick run, increase for full reproduction
    initial_gaps, final_gaps = figure1_experiment(n_mdps=50)

    print("\n" + "=" * 60)
    print("All experiments completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
