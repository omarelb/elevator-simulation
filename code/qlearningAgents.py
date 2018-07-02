import logging
import random
import pickle
import math
import os

from os.path import join

import util
import environment as env
import constants as const

from learningAgents import ReinforcementAgent

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(levelname)s:%(name)s:%(message)s')
file_handler = logging.FileHandler(join(const.LOG_DIR, 'learning.log'), mode='w')
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)


class QLearningAgent(ReinforcementAgent):
    """
    Q-Learning Agent
    """
    def __init__(self, **args):
        "You can initialize Q-values here..."
        ReinforcementAgent.__init__(self, **args)
        args['q_file'] = args['q_file'] + '_' + str(self.num_training) + '.pkl'
        if args['use_q_file'] and os.path.isfile(args['q_file']):
            with open(args['q_file'], 'rb') as q_file:
                self.episodes_so_far, self.accum_train_rewards, self.qvalues = pickle.load(q_file)
            logger.info('initialized q values from file %s', args['q_file'])
        else:
            self.qvalues = util.Counter()
            logger.info('initialized q values to 0')

    def get_qvalue(self, state, action):
        """
        Return Q(state,action).

        Should return 0.0 if we have never seen a state
        or the Q node value otherwise
        """
        return self.qvalues[(state, action)]

    def compute_value_from_qvalues(self, state):
        """
          Returns max_action Q(state,action)
          where the max is over legal actions.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return a value of 0.0.
        """
        possible_actions = (env.ElevatorState.STOP, env.ElevatorState.CONTINUE)
        return min([self.get_qvalue(state, action) for action in possible_actions])

    def compute_action_from_qvalues(self, learning_state):
        """
        Compute the best action to take in a state.
        
        Possible actions will always be a tuple (STOP, CONTINUE). Uses the boltzmann
        distribution to compute the probability of stopping.

        Parameters
        ----------
        learning_state : tuple
            The learning state is mapped to an action by the learning agent.
            It consists of, in the following order:
                - the number of up hall calls above the elevator
                - the number of down hall calls above the elevator
                - the number of up hall calls below the elevator
                - the number of down hall calls below the elevator
                - the number of car calls in the current direction of the elevator
                - current position (floor) of the elevator
                - current direction of the elevator
        """
        possible_actions = (env.ElevatorState.STOP, env.ElevatorState.CONTINUE)
        qvalues = [self.get_qvalue(learning_state, action) for action in possible_actions]
        
        # there are ties -> choose randomly
        if qvalues[0] == qvalues[1]:
            return random.choice(possible_actions)

        if qvalues[0] < qvalues[1]:
            return possible_actions[0]
        return possible_actions[1]

    def get_action(self, learning_state):
        """
        Compute the action to take in the current learning state.

        If in training, will choose actions according to boltzmann distribution on
        q values to keep exploring. Otherwise take actions with highest q value.
        """
        possible_actions = (env.ElevatorState.STOP, env.ElevatorState.CONTINUE)
        qvalues = [self.get_qvalue(learning_state, action) for action in possible_actions]

        if self.is_training:
            prob_stop = self.prob_stop(qvalues, self.temperature())
            logger.info('state:{}:qvalues(stop,continue):{}:temperature:{}:prob_stop:{}'.format(
                        learning_state, qvalues, self.temperature(), prob_stop))
            if random.random() < prob_stop:
                return env.ElevatorState.STOP
            return env.ElevatorState.CONTINUE
        return self.compute_action_from_qvalues(learning_state)

    def update(self, state, action, reward):
        """
          The parent class calls this to observe a
          state = action => next_state and reward transition.
          You should do your Q-Value update here

          NOTE: You should never call this function,
          it will be called on your behalf
        """
        util.raiseNotDefined()

    def get_policy(self, state):
        return self.compute_action_from_qvalues(state)

    def get_value(self, state):
        return self.compute_value_from_qvalues(state)

    def prob_stop(self, q_values, temperature):
        """
        Return the probability of stopping given the state's q values.

        Parameters
        ----------
        q_values : tuple
            first element is q value for stop, second element is q value for continue
        temperature : float
            the value of temperature controls the amount of randomness in the selection of actions.
            a lower temperature means a lower amount of randomness.

        Returns
        -------
        float
            probability of choosing action stop
        """
        _, prob_stop = util.boltzmann(q_values, temperature)

        return prob_stop

    def prob_continue(self, q_values, temperature):
        """
        Return the probability of choosing continue given the state's q values.

        Parameters
        ----------
        q_values : tuple
            first element is q value for stop, second element is q value for continue
        temperature : float
            the value of temperature controls the amount of randomness in the selection of actions.
            a lower temperature means a lower amount of randomness.

        Returns
        -------
        float
            probability of choosing action stop
        """
        return 1 - self.prob_stop(q_values, temperature)


class ElevatorQAgent(QLearningAgent):
    """
    Exactly the same as QLearningAgent, but with different default parameters.
    
    Parameters added
    ----------
    cost_accumulator : float:
        keeps track of elevator's cost
    decision_time : float
        time at which last decision was made
    """

    def __init__(self, beta=0.01, index=0, **args):
        """
        These default parameters can be changed from the pacman.py command line.
        For example, to change the exploration rate, try:
            python pacman.py -p PacmanQLearningAgent -a epsilon=0.1

        alpha    - learning rate
        epsilon  - exploration rate
        gamma    - discount factor
        num_training - number of training episodes, i.e. no learning after these many episodes
        """
        # args['alpha'] = alpha
        args['beta'] = beta
        self.decision_time = 0
        self.index = index
        # self.index = 0  # This is always Pacman
        QLearningAgent.__init__(self, **args)

    def get_action(self, simulator):
        """
        Simply calls the get_action method of QLearningAgent and then
        informs parent of action for Pacman.  Do not change or remove this
        method.
        """
        learning_state = simulator.environment.get_learning_state(simulator.environment.elevators[self.index])
        action = super().get_action(learning_state)
        self.do_action(learning_state, action)
        return action

    def update_accumulated_cost(self, simulator, current_event_time):
        """
        Update cost accumulator cost for elevator when some events happen.
        """
        result = 0
        t_0 = simulator.environment.last_accumulator_event_time
        t_1 = current_event_time
        d = self.decision_time
        b = self.beta
        # TODO: DO NOT UPDATE FOR PASSENGER THAT HAS JUST ARRIVED
        for passenger in simulator.environment.get_passengers_waiting():
            w_0 = passenger.waiting_time(t_0)
            w_1 = passenger.waiting_time(t_1)
            if abs(w_0) <= const.GENERAL_EPS: w_0 = 0
            if abs(w_1) <= const.GENERAL_EPS: w_1 = 0
            part_0 = math.exp(-b * (t_0 - d)) * (2 / b**3 + 2 * w_0 / b**2 + w_0**2 / b)
            part_1 = math.exp(-b * (t_1 - d)) * (2 / b**3 + 2 * w_1 / b**2 + w_1**2 / b)
            result += (part_0 - part_1) * 10e-6

        self.cost_accumulator += result
        logger.info('elevator %d cost accumulator set to %.3f', self.index, self.cost_accumulator)

    def update(self, now, next_state, reward):
        """
        The parent class calls this to observe a
        state = action => next_state and reward transition.
        You should do your Q-Value update here

        NOTE: You should never call this function,
        it will be called on your behalf
        """
        possible_actions = (env.ElevatorState.STOP, env.ElevatorState.CONTINUE)
        min_next_q = min([self.get_qvalue(next_state, action) for action in possible_actions])

        sample = reward + math.exp(-self.beta * (now - self.decision_time)) * min_next_q
        orig = self.qvalues[(self.last_state, self.last_action)]
        self.qvalues[(self.last_state, self.last_action)] = ((1 - self.alpha()) * self.get_qvalue(self.last_state, self.last_action) +
                                                             self.alpha() * sample)
        new = self.qvalues[(self.last_state, self.last_action)] 
        logger.info('time:%.2f:state:%s:action:%s:reward:%.3f:original_q:%.3f:target:%.3f:new_q:%.3f',
                    now, self.last_state, self.last_action, reward, orig, sample, new)
