"""
Visualize the vector field f(θ) for Linear DQN as shown in Figure 3 of the paper.

The vector field shows how θ evolves in different greedy regions:
- Each colored cone represents a greedy region R_ā
- The diamond markers are the landmarks (equilibrium points)
- The vector field f(θ) = b_ā - A_ā θ within each region
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
from itertools import product

from environments.MDPConstructs import create_figure2_mdp, create_figure3b_mdp, compute_babar


def compute_vector_field(P, r, Phi, epsilon, gamma, num_states, num_actions, grid_size=50):
    """
    Compute the vector field f(θ) for visualization.

    Args:
        P: Transition matrix (S, A, S)
        r: Reward matrix (S, A, S)
        Phi: Feature matrix (SA, d)
        epsilon: Exploration parameter
        gamma: Discount factor
        num_states: Number of states
        num_actions: Number of actions
        grid_size: Number of points in each dimension

    Returns:
        Grid points, vector field, policy regions, landmarks
    """
    S, A = num_states, num_actions

    # Compute b_ā and A_ā for all policies
    results = compute_babar(P, r, Phi, epsilon, gamma, S, A)

    # Compute landmarks and find bounds
    landmarks = {}
    for policy, (b_a, A_a) in results.items():
        landmark = np.linalg.solve(A_a, b_a)
        landmarks[policy] = landmark

    # Determine grid bounds based on landmarks
    all_landmarks = np.array(list(landmarks.values()))
    margin = 1.0
    x_min, x_max = all_landmarks[:, 0].min() - margin, all_landmarks[:, 0].max() + margin
    y_min, y_max = all_landmarks[:, 1].min() - margin, all_landmarks[:, 1].max() + margin

    # Create grid
    x = np.linspace(x_min, x_max, grid_size)
    y = np.linspace(y_min, y_max, grid_size)
    X, Y = np.meshgrid(x, y)

    # Compute vector field at each point
    U = np.zeros_like(X)
    V = np.zeros_like(Y)
    policy_grid = np.zeros((grid_size, grid_size), dtype=int)

    for i in range(grid_size):
        for j in range(grid_size):
            theta = np.array([X[i, j], Y[i, j]])

            # Find which greedy region this point belongs to
            Q = Phi @ theta
            Q = Q.reshape(S, A)
            policy = tuple(np.argmax(Q, axis=1))

            # Get b_ā and A_ā for this policy
            b_a, A_a = results[policy]

            # Compute f(θ) = b_ā - A_ā θ
            f_theta = b_a - A_a @ theta

            U[i, j] = f_theta[0]
            V[i, j] = f_theta[1]
            policy_grid[i, j] = policy[0] * A + policy[1]

    return X, Y, U, V, policy_grid, landmarks, results


def plot_vector_field(mdp, Phi, gamma, epsilon, title, filename):
    """
    Create vector field plot like Figure 3 in the paper.

    Args:
        mdp: Environment
        Phi: Feature matrix
        gamma: Discount factor
        epsilon: Exploration parameter
        title: Plot title
        filename: Output filename
    """
    S, A = mdp.n_states, mdp.n_actions

    print(f"Computing vector field for {title}...")
    X, Y, U, V, policy_grid, landmarks, results = compute_vector_field(
        mdp.P, mdp.r, Phi, epsilon, gamma, S, A, grid_size=40
    )

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 10))

    # Color the greedy regions
    policy_colors = {
        (0, 0): '#a8e6cf',  # Green
        (0, 1): '#ffd3b6',  # Orange
        (1, 0): '#dcedc1',  # Light green
        (1, 1): '#ffaaa5',   # Pink/Red
    }

    # Create colored regions based on policy grid
    # We'll use contourf for this
    policy_num = policy_grid.astype(float)
    colors = [policy_colors[(0, 0)], policy_colors[(0, 1)], policy_colors[(1, 0)], policy_colors[(1, 1)]]
    from matplotlib.colors import ListedColormap
    cmap = ListedColormap(colors)

    # Plot policy regions as background
    contour = ax.contourf(X, Y, policy_num, levels=[-0.5, 0.5, 1.5, 2.5, 3.5], cmap=cmap, alpha=0.3)

    # Plot vector field
    magnitude = np.sqrt(U**2 + V**2)
    magnitude_normalized = magnitude / (magnitude.max() + 1e-8)

    ax.quiver(X, Y, U, V, magnitude_normalized, cmap='viridis', alpha=0.8,
              scale=50, width=0.003, headwidth=3, headlength=4)

    # Plot landmarks
    for policy, landmark in landmarks.items():
        color = policy_colors[policy]
        ax.scatter(landmark[0], landmark[1], color='black', s=200, marker='D',
                   edgecolors='white', linewidths=2, zorder=10)
        # Label the landmark
        ax.annotate(f'{policy}', (landmark[0], landmark[1] + 0.15),
                    fontsize=9, ha='center', va='bottom')

    # Mark optimal Q* landmark (if known)
    Q_star = mdp.compute_optimal_Q(gamma)
    optimal_policy = tuple(np.argmax(Q_star, axis=1))
    if optimal_policy in landmarks:
        landmark_opt = landmarks[optimal_policy]
        ax.scatter(landmark_opt[0], landmark_opt[1], color='gold', s=300, marker='*',
                   edgecolors='black', linewidths=2, zorder=15, label=f'Optimal θ* (policy {optimal_policy})')

    ax.set_xlabel(r'$\theta_1$', fontsize=14)
    ax.set_ylabel(r'$\theta_2$', fontsize=14)
    ax.set_title(title, fontsize=16)
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)

    # Add text explaining regions
    textstr = '\n'.join([
        f'Policy (a₀, a₁):',
        f'  (0,0): Green (top-left)',
        f'  (0,1): Orange',
        f'  (1,0): Light green',
        f'  (1,1): Red (bottom-right)',
        f'\nDiamonds: Landmarks (equilibria)'
    ])
    ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=9,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.savefig(filename.replace('.png', '.pdf'), bbox_inches='tight')
    print(f"Saved {filename}")
    plt.close()

    return landmarks, results


def analyze_landmarks(landmarks, results, mdp, Phi, gamma):
    """
    Analyze the landmarks to determine which policies they correspond to.

    For each landmark, compute:
    1. The greedy policy at that landmark
    2. The value of that policy
    """
    S, A = mdp.n_states, mdp.n_actions

    print("\nLandmark Analysis:")
    print("-" * 60)

    # Compute optimal Q* for reference
    Q_star = mdp.compute_optimal_Q(gamma)
    V_star = np.max(Q_star, axis=1)
    optimal_policy = np.argmax(Q_star, axis=1)

    print(f"Optimal policy: {tuple(optimal_policy)}")
    print(f"Optimal V*: {V_star}")
    print()

    for policy, landmark in landmarks.items():
        # Verify that the landmark is in the correct greedy region
        Q_landmark = Phi @ landmark
        Q_landmark = Q_landmark.reshape(S, A)
        greedy_at_landmark = tuple(np.argmax(Q_landmark, axis=1))

        # Compute value of this policy
        policy_arr = np.array(policy)
        P_pi = np.zeros((S, S))
        r_pi = np.zeros(S)
        for s in range(S):
            P_pi[s] = mdp.P[s, policy_arr[s]]
            r_pi[s] = np.sum(mdp.P[s, policy_arr[s]] * mdp.r[s, policy_arr[s]])

        V_policy = np.linalg.solve(np.eye(S) - gamma * P_pi, r_pi)
        mean_V = V_policy.mean()

        print(f"Policy {policy}:")
        print(f"  Landmark: θ = [{landmark[0]:.4f}, {landmark[1]:.4f}]")
        print(f"  Greedy at landmark: {greedy_at_landmark}")
        print(f"  Value V^π: {V_policy}")
        print(f"  Mean value: {mean_V:.4f}")
        print(f"  Gap from optimal: {np.max(np.abs(V_star - V_policy)):.4f}")
        print()


def main():
    """Generate vector field plots for both MDPs."""
    print("=" * 60)
    print("Vector Field Visualization")
    print("=" * 60)

    # Figure 2 MDP
    print("\n" + "=" * 60)
    print("FIGURE 2 MDP: Three trajectory behaviors")
    print("=" * 60)
    mdp, Phi, gamma, epsilon = create_figure2_mdp()
    landmarks_fig2, results_fig2 = plot_vector_field(
        mdp, Phi, gamma, epsilon,
        "Figure 2 MDP: Vector Field f(θ)",
        "figure2_vector_field.png"
    )
    analyze_landmarks(landmarks_fig2, results_fig2, mdp, Phi, gamma)

    # Figure 3b MDP
    print("\n" + "=" * 60)
    print("FIGURE 3b MDP: Convergence to worst policy")
    print("=" * 60)
    mdp, Phi, gamma, epsilon = create_figure3b_mdp()
    landmarks_fig3b, results_fig3b = plot_vector_field(
        mdp, Phi, gamma, epsilon,
        "Figure 3b MDP: Vector Field f(θ) (Worst Policy)",
        "figure3b_vector_field.png"
    )
    analyze_landmarks(landmarks_fig3b, results_fig3b, mdp, Phi, gamma)

    print("\n" + "=" * 60)
    print("Vector field visualization complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()