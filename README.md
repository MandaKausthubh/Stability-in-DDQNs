# Linear DQN Reproduction: "Does DQN Learn?"

This repository reproduces the experiments from the paper **"Does DQN Learn?"** by Aditya Gopalan and Gugan Thoppe (IEEE Transactions on Automatic Control, 2025 / arXiv:2205.13617).

## Paper Summary

The paper provides a theoretical analysis showing that Deep Q-Networks (DQN) can fail to meet even basic learning criteria. Specifically:

1. **Main Finding**: DQN can produce policies worse than the initial guess, even when all state-action pairs are visited infinitely often (a condition that guarantees convergence for tabular Q-learning).

2. **Theoretical Framework**: The authors use differential inclusion theory to analyze linear DQN (DQN with linear function approximation, keeping ε-greedy exploration, experience replay, and target networks).

3. **Key Results**:
   - Limit points of linear DQN correspond to fixed points of projected Bellman operators
   - These fixed points need not relate to optimal—or even near-optimal—policies
   - Specific MDP constructions where linear DQN always converges to the worst policy

## Theoretical Background

### Linear DQN Algorithm

The paper studies linear DQN with Q-value approximation:
```
Q(s, a) = φ(s, a)^T θ
```

The update rule follows:
```
θ_{n+1} = θ_n + α_n δ_n φ(s_n, a_n)
δ_n = r(s_n, a_n, s'_n) + γ φ(s'_n, a'_n)^T θ_n^- - φ(s_n, a_n)^T θ_n
```

Where:
- `θ_n`: current parameter estimate
- `θ_n^-`: target network parameter (updated on faster timescale)
- `α_n`: step size (paper uses `α_n = log(n) / (10n)` or `α_n = 2/n`)
- `γ`: discount factor
- `ε`: exploration parameter for ε-greedy policy

### Key Equations

For a deterministic policy ā, the paper defines:
- **b_ā** = E[φ(s,a) r(s,a,s')] = Φ^T D_ā^ε r
- **A_ā** = E[φ(s,a) φ(s,a)^T - γ φ(s,a) φ(s',a')^T] = Φ^T D_ā^ε (I - γ P_ā^{ε'}) Φ

The vector field within each greedy region R_ā is:
```
f(θ) = b_ā - A_ā θ
```

The landmark (equilibrium point) for each region is `θ* = A_ā^{-1} b_ā`.

## Repository Structure

```
├── environments/
│   ├── DiscreteMDPs.py      # Base MDP class
│   └── MDPConstructs.py     # Paper's specific MDP constructions
├── models/
│   └── model.py              # Neural network feature extractor (for experiments)
├── utils/
│   ├── Sampling.py           # Stationary distribution computations
│   └── ReplayBuffer.py       # Experience replay buffer
├── LinearDQN.py             # Main Linear DQN implementation
├── run_experiments.py        # Reproduce paper figures
├── visualize_vector_field.py # Visualize dynamics in parameter space
├── Experiments.py           # Original experiment code
├── DDQN.py                  # Double DQN extension
└── DoesDQNLearn.pdf         # Paper PDF
```

## MDP Constructions

### Figure 2 MDP (Three trajectory behaviors)

2-state, 2-action MDP with:
- Transition matrices:
  ```
  P(:,:,1) = [0.380, 0.620; 0.786, 0.214]
  P(:,:,2) = [0.124, 0.876; 0.426, 0.574]
  ```
- Expected rewards: r = [-0.031, 0.785; -0.282, -0.418]
- Discount factor: γ = 0.9
- Exploration: ε = 0.05

Feature matrix Φ (4×2):
```
Φ = [1.919, 0.112;
     2.581, -0.659;
     1.912, 1.679;
     1.560, -0.168]
```

This MDP demonstrates three behaviors:
- **Blue trajectory**: Converges to optimal policy
- **Green trajectory**: Converges to sub-optimal policy
- **Red trajectory**: Policy oscillation (sliding mode attractor)

### Figure 3b MDP (Worst policy convergence)

2-state, 2-action MDP where linear DQN always converges to the worst policy:
- Transition matrices:
  ```
  P(:,:,1) = [0.355, 0.645; 0.598, 0.402]
  P(:,:,2) = [0.820, 0.180; 0.288, 0.712]
  ```
- Expected rewards: r = [-0.599, -1.427; 0.658, 0.300]
- Discount factor: γ = 0.75
- Exploration: ε = 0.1

## Installation

```bash
pip install -r requirements.txt
```

## Running Experiments

```bash
# Reproduce all paper figures
python run_experiments.py

# Visualize vector fields
python visualize_vector_field.py
```

## Key Findings

1. **Invariant Sets**: Linear DQN converges to invariant sets of the limiting differential inclusion, not necessarily to optimal policies.

2. **Sliding Modes**: On boundaries between greedy regions, the dynamics can exhibit sliding mode behavior, leading to policy oscillation.

3. **No Lyapunov Function**: Unlike tabular Q-learning, linear DQN lacks a global Lyapunov function, allowing convergence to arbitrary invariant sets.

4. **Worst Policy Scenario**: The paper constructs an MDP where the only stable equilibrium corresponds to the worst possible policy.

## References

- Paper: "Does DQN Learn?" by Aditya Gopalan and Gugan Thoppe
  - arXiv: https://arxiv.org/abs/2205.13617
  - IEEE TAC: https://doi.org/10.1109/TAC.2025.3631342

## License

MIT License