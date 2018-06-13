"""
Defines elevator agents that are rule-based controllers. 
"""

from learningAgents import Agent

import util
import environment as env
import numpy.random as rnd


class HeuristicAgent(Agent):
    pass


class BestFirstAgent(HeuristicAgent):
    """
    Agent which returns actions, either stop or continue.

    Always picks up the first awaiting call in the current moving direction
    to be served.
    """
    def get_action(self, simulator):
        """
        Called by elevator when a control decision is necessary.
        """
        elevator_state = simulator.environment.elevators[self.index]
        next_floor = simulator.environment.floors[elevator_state.stop_target]
        num_up = next_floor.num_up()
        num_down = next_floor.num_down()
        if elevator_state.direction == env.ElevatorState.UP:
            if num_up > 0:
                return env.ElevatorState.STOP
            elif num_down > 0 and not simulator.environment.is_hall_call(elevator_state.stop_target, down=False, above=True):
                return env.ElevatorState.STOP
        if elevator_state.direction == env.ElevatorState.DOWN:
            if num_down > 0:
                return env.ElevatorState.STOP
            elif num_up > 0 and not simulator.environment.is_hall_call(elevator_state.stop_target, down=True, above=False):
                return env.ElevatorState.STOP

        return env.ElevatorState.CONTINUE


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
        if rnd.rand() < prob_cont:
            return env.ElevatorState.CONTINUE
        return env.ElevatorState.STOP

    def __repr__(self):
        return 'RandomAgent()'
