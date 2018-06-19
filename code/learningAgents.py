# learningAgents.py
# -----------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).

# from game import Directions, Agent, Actions

import time
import math
import pickle
import os
import csv

from os.path import join

import util
import constants as const


class Agent:
    """
    An agent must define a get_action method, but may also define the
    following methods which will be called if they exist:

    Attributes
    ----------
    index : int
        agent identifier
    decision_time : float
        simulated time at which a decision was last made
    cost_accumulator : float
        cost the agent has accumulated from last decision time until now
    """
    def __init__(self, index=0):
        self.index = index
        self.decision_time = 0
        # cost accumulator for an elevator.
        self.cost_accumulator = 0

    def get_action(self, simulator):
        """
        The Agent will receive a GameState (from either {pacman, capture, sonar}.py) and
        must return an action from Directions.{North, South, East, West, Stop}
        """
        util.raiseNotDefined()

    def start_episode(self):
        """
          Called by environment when new episode is starting
        """
        self.last_state = None
        self.last_action = None
        self.episode_rewards = 0.0


class ValueEstimationAgent(Agent):
    """
    Abstract agent which assigns values to (state,action)
    Q-Values for an environment. As well as a value to a
    state and a policy given respectively by,

    V(s) = max_{a in actions} Q(s,a)
    policy(s) = arg_max_{a in actions} Q(s,a)

    Both ValueIterationAgent and QLearningAgent inherit
    from this agent. While a ValueIterationAgent has
    a model of the environment via a MarkovDecisionProcess
    (see mdp.py) that is used to estimate Q-Values before
    ever actually acting, the QLearningAgent estimates
    Q-Values while acting in the environment.
    """
    def __init__(self, beta=0.01, num_training=10):
        """
        Parameters
        ----------
        alpha : float
            learning rate
        beta : float
            continuous analog of discounting factor gamma
        temperature : float
            controls randomness of action selection
        num_training : int
            number of training episodes, i.e. no learning after these many episodes
        """
        # self.alpha = float(alpha)
        self.beta = float(beta)
        self.num_training = int(num_training)

    ####################################
    #    Override These Functions      #
    ####################################
    def get_qvalue(self, state, action):
        """
        Should return Q(state,action)
        """
        util.raiseNotDefined()

    def get_value(self, state):
        """
        What is the value of this state under the best action?
        Concretely, this is given by

        V(s) = max_{a in actions} Q(s,a)
        """
        util.raiseNotDefined()

    def get_policy(self, state):
        """
        What is the best action to take in the state. Note that because
        we might want to explore, this might not coincide with get_action
        Concretely, this is given by

        policy(s) = arg_max_{a in actions} Q(s,a)

        If many actions achieve the maximal Q-value,
        it doesn't matter which is selected.
        """
        util.raiseNotDefined()

    def get_action(self, state):
        """
        state: can call state.get_legal_actions()
        Choose an action and return it.
        """
        util.raiseNotDefined()


class ReinforcementAgent(ValueEstimationAgent):
    """
    Abstract Reinforcement Agent: A ValueEstimationAgent
        which estimates Q-Values (as well as policies) from experience
        rather than a model

    The environment will call
    observe_transition(state,action,next_state,delta_reward),
    which will call update(state, action, next_state, delta_reward)
    which you should override.

    Use self.get_legal_actions(state) to know which actions are available in a state
    """
    def __init__(self, beta=0.01, **args):
        """
        Parameters
        ----------
        alpha : float
            learning rate
        beta : float
            continuous analog of discounting factor gamma
        num_training : int
            number of training episodes, i.e. no learning after these many episodes
        """
        temperature_end = 0.01
        self.annealing_factor = args['annealing_factor']
        self.is_training = args['is_training']
        self.num_training = int(math.log(temperature_end / 2, self.annealing_factor))
        self.episodes_so_far = 0
        self.accum_train_rewards = 0.0
        self.accum_test_rewards = 0.0
        # self.start_alpha = float(start_alpha)
        self.beta = float(beta)
        self.use_q_file = args['use_q_file']
        args['q_file'] = args['q_file'] + '_' + str(self.num_training) + '.pkl'
        self.q_file = args['q_file']
        self.qvalues = None

    def update(self, state, action, delta_reward):
        """
        This class will call this function after observing a transition and reward.
        """
        util.raiseNotDefined()

    def observe_transition(self, observed_time, action, delta_reward):
        """
        Called by environment to inform agent that a transition has
        been observed. This will result in a call to self.update
        on the same arguments
        """
        self.episode_rewards += delta_reward
        self.update(observed_time, action, delta_reward)

    def stop_episode(self):
        """
        Called by environment when episode is done.
        """
        if self.is_training:
            self.accum_train_rewards += self.episode_rewards
        else:
            self.accum_test_rewards += self.episode_rewards
        self.episodes_so_far += 1
        # TODO: IF MULTIPLE ELEVATORS, MAKE SURE THIS ONLY GETS CALLED ONCE

    def temperature(self):
        """
        Return boltzmann temperature as a function of which episode of training we're in.
        """
        return 2 * self.annealing_factor**self.episodes_so_far

    def alpha(self):
        """
        Return update step-size as a function of which episode of training we're in.
        """
        # TODO: MAYBE CHANGE 0.01 TO START_ALPHA PARAMETER
        return 0.01 * 0.99975**self.episodes_so_far

    def do_action(self, state, action):
        """
            Called by inherited class when
            an action is taken in a state
        """
        self.last_state = state
        self.last_action = action

    def register_initial_state(self, state):
        self.start_episode()
        if self.episodes_so_far == 0:
            print('Beginning %d episodes of Training' % (self.num_training))

    def final(self, passenger_statistics):
        """
        Called by environment at the terminal state

        Parameters
        ----------
        passenger_statistics : list
            containing tuples (waiting_time, boarding_time, system_time, waiting_time > 60)
            for each passenger in the simulation. Averages need to be written to file.
        """
        append_name = 'train' if self.is_training else 'test'
        self.stop_episode()
        if self.use_q_file and self.is_training:
            with open(self.q_file, 'wb') as q_file:
                pickle.dump((self.episodes_so_far, self.accum_train_rewards, self.qvalues), q_file)
        passenger_datafile = join('data', 'passenger_statistics_{}_{}.csv'.format(append_name, self.num_training))
        episode_rewards_file = join('data', 'episode_rewards_{}_{}.csv'.format(append_name, self.num_training))
        # TODO: MAKE STATISTICS FILENAME VARIABLE
        with open(passenger_datafile, 'a') as f:
            if not os.path.isfile(passenger_datafile):
                f.write('episode,waiting_time,boarding_time,system_time,threshold\r\n')
            avg_waiting = sum([x[0] for x in passenger_statistics]) / len(passenger_statistics)
            avg_boarding = sum([x[1] for x in passenger_statistics]) / len(passenger_statistics)
            avg_system = sum([x[2] for x in passenger_statistics]) / len(passenger_statistics)
            avg_threshold = sum([x[3] for x in passenger_statistics]) / len(passenger_statistics)
            csv_writer = csv.writer(f)
            csv_writer.writerow((self.episodes_so_far, avg_waiting, avg_boarding, avg_system, avg_threshold))
        # TODO: Parameterize data_dir and csv filename depending on training, etc.

        with open(episode_rewards_file, 'a') as f:
            if not os.path.isfile(episode_rewards_file):
                f.write('episode,cost\r\n')
            csv_writer = csv.writer(f)
            csv_writer.writerow((self.episodes_so_far, self.episode_rewards))
        # Make sure we have this var
        if 'episode_start_time' not in self.__dict__:
            self.episode_start_time = time.time()
        if 'last_window_accum_rewards' not in self.__dict__:
            self.last_window_accum_rewards = 0.0
        
        # TODO: REWRITE TO INCORPORATE MULTIPLE AGENTS
        self.last_window_accum_rewards += self.episode_rewards

        if self.episodes_so_far % const.NUM_EPS_UPDATE == 0:
            print('Reinforcement Learning Status:')
            window_avg = self.last_window_accum_rewards / const.NUM_EPS_UPDATE
            if self.episodes_so_far <= self.num_training:
                train_avg = self.accum_train_rewards / self.episodes_so_far
                print('\tCompleted %d out of %d training episodes' % (
                      self.episodes_so_far, self.num_training))
                print('\tAverage Rewards over all training: %.2f' % (
                      train_avg))
            else:
                test_avg = float(self.accum_test_rewards) / (self.episodes_so_far - self.num_training)
                print('\tCompleted %d test episodes' % (self.episodes_so_far - self.num_training))
                print('\tAverage Rewards over testing: %.2f' % test_avg)
            print('\tAverage Rewards for last %d episodes: %.2f' % (
                  const.NUM_EPS_UPDATE, window_avg))
            print('\tEpisode took %.2f seconds' % (time.time() - self.episode_start_time))
            self.last_window_accum_rewards = 0.0
            self.episode_start_time = time.time()

        if self.episodes_so_far == self.num_training:
            msg = 'Training Done (turning off epsilon and alpha)'
            print('%s\n%s' % (msg, '-' * len(msg)))
