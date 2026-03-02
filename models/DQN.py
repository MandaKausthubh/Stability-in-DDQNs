from .DQN import DQN

class DQNAgent:
    """
    DQN agent with a general function approximation architecture for Q(s, a).
    """
    def __init__(
        self,
        environment,
        num_states,
        num_actions,
    )
