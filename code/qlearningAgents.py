# qlearningAgents.py
# ------------------
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


# from game import *
from learningAgents import ReinforcementAgent
# from featureExtractors import *

import random, util, math

class QLearningAgent(ReinforcementAgent):
    """
      Q-Learning Agent

      Functions you should fill in:
        - compute_value_from_qvalues
        - compute_action_from_qvalues
        - get_qvalue
        - get_action
        - update

      Instance variables you have access to
        - self.epsilon (exploration prob)
        - self.alpha (learning rate)
        - self.discount (discount rate)

      Functions you should use
        - self.get_legal_actions(state)
          which returns legal actions for a state
    """
    def __init__(self, **args):
        "You can initialize Q-values here..."
        ReinforcementAgent.__init__(self, **args)

        # initialize them all to zero
        # qvalues can be accessed through self.qvalues[(state, action)]
        self.qvalues = util.Counter()


    def get_qvalue(self, state, action):
        """
          Returns Q(state,action)
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
        possible_actions = self.get_legal_actions(state)
        # if there are legal actions
        if possible_actions:
            return max([self.get_qvalue(state, action) for action in possible_actions])
        # reached terminal state, return 0
        return 0

    def compute_action_from_qvalues(self, state):
        """
          Compute the best action to take in a state.  Note that if there
          are no legal actions, which is the case at the terminal state,
          you should return None.
        """
        possible_actions = self.get_legal_actions(state)
        # reached terminal state
        if not possible_actions:
            return None

        qvalues = [self.get_qvalue(state, action) for action in possible_actions]
        max_qvalue = max(qvalues)

        # there are ties
        if qvalues.count(max_qvalue) > 1:
            tied = []

            for action, qvalue in zip(possible_actions, qvalues):
                if qvalue == max_qvalue:
                    # append all tied actions to a list for choosing randomly later
                    tied.append((action, max_qvalue))

            # choose randomly from tied values
            action, _ = random.choice(tied)

            return action
        
        # there were no ties, return action that yields maximal q value
        argmax_ix = qvalues.index(max(qvalues))
        return possible_actions[argmax_ix]

    def get_action(self, state):
        """
          Compute the action to take in the current state.  With
          probability self.epsilon, we should take a random action and
          take the best policy action otherwise.  Note that if there are
          no legal actions, which is the case at the terminal state, you
          should choose None as the action.

          HINT: You might want to use util.flip_coin(prob)
          HINT: To pick randomly from a list, use random.choice(list)
        """
        # Pick Action
        legal_actions = self.get_legal_actions(state)
        # terminal state
        if not legal_actions:
            return None

        # choose an action randomly with probability epsilon
        if util.flip_coin(self.epsilon):
            # choose an action randomly
            return random.choice(legal_actions)
        return self.compute_action_from_qvalues(state)


    def update(self, state, action, next_state, reward):
        """
          The parent class calls this to observe a
          state = action => next_state and reward transition.
          You should do your Q-Value update here

          NOTE: You should never call this function,
          it will be called on your behalf
        """
        "*** YOUR CODE HERE ***"
        next_legal_actions = self.get_legal_actions(next_state)

        # if no next legal actions, we have run into terminal state, nextQ should be 0
        max_next_q = 0
        if next_legal_actions:
            max_next_q = max([self.get_qvalue(next_state, next_action) for next_action in next_legal_actions])

        sample = reward + self.discount * max_next_q

        self.qvalues[(state, action)] = (1 - self.alpha) * self.get_qvalue(state, action) + self.alpha * sample


    def get_policy(self, state):
        return self.compute_action_from_qvalues(state)

    def get_value(self, state):
        return self.compute_value_from_qvalues(state)


class PacmanQAgent(QLearningAgent):
    "Exactly the same as QLearningAgent, but with different default parameters"

    def __init__(self, epsilon=0.05,gamma=0.8,alpha=0.2, num_training=0, **args):
        """
        These default parameters can be changed from the pacman.py command line.
        For example, to change the exploration rate, try:
            python pacman.py -p PacmanQLearningAgent -a epsilon=0.1

        alpha    - learning rate
        epsilon  - exploration rate
        gamma    - discount factor
        num_training - number of training episodes, i.e. no learning after these many episodes
        """
        args['epsilon'] = epsilon
        args['gamma'] = gamma
        args['alpha'] = alpha
        args['num_training'] = num_training
        self.index = 0  # This is always Pacman
        QLearningAgent.__init__(self, **args)

    def get_action(self, state):
        """
        Simply calls the get_action method of QLearningAgent and then
        informs parent of action for Pacman.  Do not change or remove this
        method.
        """
        action = QLearningAgent.get_action(self,state)
        self.do_action(state, action)
        return action


# class ApproximateQAgent(PacmanQAgent):
#     """
#        ApproximateQLearningAgent

#        You should only have to overwrite get_qvalue
#        and update.  All other QLearningAgent functions
#        should work as is.
#     """
#     def __init__(self, extractor='IdentityExtractor', **args):
#         self.featExtractor = util.lookup(extractor, globals())()
#         PacmanQAgent.__init__(self, **args)
#         self.weights = util.Counter()

#     def getWeights(self):
#         return self.weights

#     def get_qvalue(self, state, action):
#         """
#           Should return Q(state,action) = w * featureVector
#           where * is the dotProduct operator
#         """
#         "*** YOUR CODE HERE ***"
#         result = 0
#         featureVector = self.featExtractor.getFeatures(state, action)
#         weights = self.getWeights()
#         for key, value in featureVector.iteritems():
#             # print key, value
#             result += weights[key] * value

#         # product of two counters returns dot product
#         # print result
#         return result


#     def update(self, state, action, next_state, reward):
#         """
#            Should update your weights based on transition
#         """
#         next_legal_actions = self.get_legal_actions(next_state)
        
#         # print self.featExtractor.getFeatures(state, action)
#         # exit()
#         # if no next legal actions, we have run into terminal state, nextQ should be 0
#         max_next_q = 0
#         if next_legal_actions:
#             nextQ = ([self.get_qvalue(next_state, next_action) for next_action in next_legal_actions])
#             max_next_q = max(nextQ)
#             # max_next_q = max([self.get_qvalue(next_state, next_action) for next_action in next_legal_actions])

#         difference = reward + self.discount * max_next_q - self.get_qvalue(state, action)
#         features = self.featExtractor.getFeatures(state, action).copy()
#         for key, value in features.iteritems():
#             # print key, value
#             self.weights[key] += self.alpha * difference * value
#         # print "legal_actions: {}\nreward: {}\ndiscount: {}\nalpha: {}\nQ(s, a): {}\nQ(s',a'): {}\nfeatures: {}\nweights: {}".format(
#         #     next_legal_actions, reward, self.discount, self.alpha, self.get_qvalue(state, action), max_next_q,
#         #     self.featExtractor.getFeatures(state, action), self.getWeights()
#         # )
#         # print 'difference: ', difference



#     def final(self, state):
#         "Called at the end of each game."
#         # call the super-class final method
#         PacmanQAgent.final(self, state)

#         # did we finish training?
#         if self.episodes_so_far == self.num_training:
#             # you might want to print your weights here for debugging
#             "*** YOUR CODE HERE ***"
#             print(self.getWeights())
#             pass