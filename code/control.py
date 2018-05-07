"""
This module defines agents that determine the behaviour of elevators.
"""

from abc import ABC, abstractmethod

class Agent:
    """
    Base class for an agent.

    An agent controls the actions an elevator takes.
    """
    def __init__(self):
        pass

    @abstractmethod
    def get_action(self, state):
        """
        Return an action the agent should execute.

        Parameters
        ----------
        state :
            state of the environment (according to API, TO BE DEFINED)
        
        Returns
        -------
        action :
            action to be executed by agent requesting the action (according to API, TO BE DEFINED)
        """
        pass


class ReinforcementAgent(Agent):
    """
    Implements a reinforcement learning agent.
    """
    def __init__(self):
        pass

    
    def get_action(self, state):
        pass


class OtherAlgorithmAgent:
    pass