"""
Defines elevator agents that are rule-based controllers. 
"""

from learningAgents import Agent

import util
import environment

class HeuristicAgent(Agent):
    pass


class FirstComeAgent(HeuristicAgent):
    pass


class RandomAgent(Agent):
    """
    Agent which returns actions, either stop or continue, completely at random.

    Parameters
    ----------
    simulator :
        simulator
    prob_cont : float
        float \in [0, 1] indicating the probability of choosing action continue
    """
    def get_action(self, simulator, prob_cont=0.5):
        if util.flip_coin(prob_cont):
            return environment.ElevatorState.CONTINUE
        return environment.ElevatorState.STOP
