from multibot_cluster_env import MultiBotClusterEnv

class SwarmEnvironment(MultiBotClusterEnv):
    """
    Enhanced swarm environment that extends MultiBotClusterEnv
    with decentralized control and visualization capabilities
    """
    def __init__(self, num_bots=3, dt=0.05, T=10.0, task="translate"):
        super().__init__(num_bots=num_bots, dt=dt, T=T, task=task) 