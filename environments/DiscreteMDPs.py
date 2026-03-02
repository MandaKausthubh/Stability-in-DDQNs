import gym
import numpy as np
from gym import spaces


class DiscreteMDP(gym.Env):
    """Finite MDP that follows gym interface"""
    metadata = {'render.modes': ['human']}

    def __init__(
        self,
        n_states, # num states S
        n_actions, # num actions A
        P, # prob transition matrix [S, A, S], numpy.ndarray
        r, # reward vector [S, A], numpy.ndarray
        rho=None
    ): # initial state distribution, numpy.ndarray
        super().__init__()
        self.action_space = spaces.Discrete(n_actions)
        # Example for using image as input (channel-first; channel-last also works):
        self.observation_space = spaces.Discrete(n_states)
        self.n_actions = n_actions
        self.n_states = n_states

        self.P = P
        self.r = r
        self.rho = rho
        if not self.rho:
            self.rho = np.ones(n_states)/n_states
        self.state = 0


    def step(self, action):   # type: ignore
        observation = self.np_random.choice(
            self.observation_space.n,   # type: ignore
            p=self.P[self.state, action]
        )

        reward = self.r[self.state, action]
        terminated = False
        truncated = False
        info = {}

        self.state = observation
        return observation, reward, terminated, truncated, info


    def reset(self, seed=None, options=None):   # type: ignore
        super().reset(seed=seed)
        self.state = self.np_random.choice(
            self.observation_space.n,   # type: ignore
            p=self.rho
        )
        return self.state, {}


    def render(self, mode='human'):
        pass


    def close (self):
        pass
